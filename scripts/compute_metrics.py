#############################################
#
#
#   Code to do post-hoc analysis on a directory of checkpoints
#
#
#############################################

import datetime
import os

from absl import app, flags, logging
from flax.traverse_util import flatten_dict
import jax
from ml_collections import config_flags
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb

from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils import filter_eval_datasets, process_text, TrainState

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoints",
    None,
    "Path to directory of checkpoints to visualize. ",
)
flags.DEFINE_integer(
    "eval_every", None, "If not None, skip any steps not divisible by eval_every."
)
flags.DEFINE_string(
    "name",
    "evaluation",
    "Wandb name. ",
)


# config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(
        "/nfs/nfs1/users/riadoshi/orca/experiments/homer/configs/",
        "gnm_datasep_config.py:vit_s",
    ),
    "File path used to get the dataset kwargs.",
    lock_config=False,
)


def main(_):
    initialize_compilation_cache()

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    model = OctoModel.load_pretrained(FLAGS.checkpoints)
    text_processor = model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        def get_ws_embeddings(embeddings, ws):
            return {
                k: (
                    v
                    if v.tokens.ndim == 3
                    else v.replace(tokens=v.tokens[:, :ws], mask=v.mask[:, :ws])
                )
                for k, v in embeddings.items()
            }

        all_metrics = {}
        for ws in range(1, batch["observation"]["timestep_pad_mask"].shape[1] + 1):
            ws_embeddings = get_ws_embeddings(transformer_embeddings, ws)
            _, ws_action_metrics = bound_module.heads["action"].loss(
                ws_embeddings,  # action head knows to pull out the "action" readout_key
                batch["action"][:, :ws],
                batch["observation"]["timestep_pad_mask"][:, :ws],
                batch["action_pad_mask"][:, :ws],
                train=train,
            )

            predicted_actions = bound_module.heads["action"].predict_action(
                ws_embeddings, rng=rng, train=False
            )

            ws_action_metrics["samples_mse"] = (
                ((batch["action"][:, ws - 1, 0] - predicted_actions[:, 0]) ** 2)
                .sum(-1)
                .mean()
            )
            all_metrics[f"ws_{ws}"] = ws_action_metrics
        return 0, all_metrics

    # load datasets
    if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
        # create dataset_kwargs_list from oxe_kwargs
        (
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            FLAGS.config.dataset_kwargs["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.config.dataset_kwargs["oxe_kwargs"]
        )
        del FLAGS.config.dataset_kwargs["oxe_kwargs"]

    val_datasets_kwargs_list, _ = filter_eval_datasets(
        FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
        FLAGS.config.dataset_kwargs["sample_weights"],
        FLAGS.config.eval_datasets,
    )
    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=lambda batch: process_batch(batch),
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset_kwargs=FLAGS.config.dataset_kwargs,
        **FLAGS.config.val_kwargs.to_dict(),
    )
    train_val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=lambda batch: process_batch(batch),
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset_kwargs=FLAGS.config.dataset_kwargs,
        train=True,
        **FLAGS.config.val_kwargs.to_dict(),
    )
    viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset_kwargs=FLAGS.config.dataset_kwargs,
        **FLAGS.config.viz_kwargs.to_dict(),
    )
    train_viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset_kwargs=FLAGS.config.dataset_kwargs,
        train=True,
        **FLAGS.config.viz_kwargs.to_dict(),
    )
    if "rollout_kwargs" in FLAGS.config:
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            history_length=FLAGS.config["window_size"],
            model_pred_horizon=FLAGS.config["model"]["heads"]["action"]["kwargs"].get(
                "pred_horizon", 1
            ),
            **FLAGS.config.rollout_kwargs.to_dict(),
        )
        print(rollout_callback)
    else:
        rollout_callback = None

    list_of_checkpoints = ocp.utils.checkpoint_steps_paths(FLAGS.checkpoints)
    list_of_checkpoints = sorted(
        list_of_checkpoints,
        key=lambda path: ocp.utils.step_from_checkpoint_name(path.name),
    )
    logging.info(list_of_checkpoints)

    wandb_id = "{name}_{time}".format(
        name=FLAGS.name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        id=wandb_id,
    )

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    checkpointer = ocp.PyTreeCheckpointer()

    for path in list_of_checkpoints:
        step = ocp.utils.step_from_checkpoint_name(path.name)
        if FLAGS.eval_every is not None and step % FLAGS.eval_every != 0:
            continue
        print(f"Loading checkpoint {step}: ", path)
        params = checkpointer.restore(tf.io.gfile.join(path, "default"), model.params)
        model = model.replace(params=params)

        train_state = TrainState.create(
            rng=jax.random.PRNGKey(1234),
            model=model,
            tx=optax.adamw(optax.constant_schedule(0.0)),  # dummy optimizer
        )

        # validation metrics
        val_metrics = val_callback(train_state, step)
        wandb_log(val_metrics, step=step)

        # visualizations
        viz_metrics = viz_callback(train_state, step)
        wandb_log(viz_metrics, step=step)

        train_val_metrics = train_val_callback(train_state, step)
        wandb_log({"train_dataset": train_val_metrics}, step=step)

        # visualizations
        train_viz_metrics = train_viz_callback(train_state, step)
        wandb_log({"train_dataset": train_viz_metrics}, step=step)

        # optional: rollout eval
        if rollout_callback is not None:
            rollout_metrics = rollout_callback(train_state, step)
            wandb_log(rollout_metrics, step=step)


if __name__ == "__main__":
    app.run(main)
