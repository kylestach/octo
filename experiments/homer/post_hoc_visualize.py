#############################################
#
#
#   Code to do post-hoc analysis on a directory of checkpoints
#
#
#############################################

import datetime
import os
from copy import deepcopy

from absl import app, flags, logging
from flax.traverse_util import flatten_dict
import jax
from ml_collections import config_flags
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb

from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils import filter_eval_datasets, process_text, TrainState
from eval_utils import download_checkpoint_from_gcs
import experiments.homer.sim.wrappers.go1_wrapper

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoints",
    None,
    "Path to directory of checkpoints to visualize. ",
)
flags.DEFINE_integer("step", None, "If not None, eval at this step.")
flags.DEFINE_integer(
    "eval_every", None, "If not None, eval all steps divisible by eval_every."
)
flags.DEFINE_string(
    "name",
    None,
    "Wandb name. ",
)
flags.DEFINE_string(
    "checkpoint_cache_dir", "/tmp/", "Where to cache checkpoints downloaded from GCS"
)


config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py:gc_bridge"),
    "File path used to get the dataset kwargs.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "update_config",
    os.path.join(config_dir, "config.py:gc_bridge"),
    "File path used to get the dataset kwargs.",
    lock_config=False,
)


def main(_):
    initialize_compilation_cache()

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # checkpointer = ocp.CheckpointManager(
    #     FLAGS.checkpoints, ocp.PyTreeCheckpointer()
    # )

    if FLAGS.step is None:
        weights_path, step = download_checkpoint_from_gcs(
            FLAGS.checkpoints,
            checkpointer.latest_step(),
            FLAGS.checkpoint_cache_dir,
        )
    else:
        weights_path, step = download_checkpoint_from_gcs(
            FLAGS.checkpoints,
            FLAGS.step,
            FLAGS.checkpoint_cache_dir,
        )
    model = OctoModel.load_pretrained(weights_path, int(step))

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

        # add multi-head loss support
        action_loss, action_metrics = 0, {}
        for head_name, head in bound_module.heads.items():
            head_loss, head_metrics = head.loss(
                transformer_embeddings,  # action head knows to pull out the "action" readout_key
                batch["action"],
                batch["observation"]["timestep_pad_mask"],
                batch["action_pad_mask"],
                action_head_mask=batch["action_head_masks"][head_name],
                train=train,
            )

            # weight loss by number of samples from each head
            head_sample_fraction = (batch["action_head_masks"][head_name].sum()) / len(
                batch["action"]
            )
            action_loss += head_loss * head_sample_fraction * head.loss_weight
            action_metrics[head_name] = head_metrics
        return action_loss, action_metrics

    config = deepcopy(FLAGS.config)
    config.update(FLAGS.update_config)

    # val_datasets_kwargs_list, _ = filter_eval_datasets(
    #     config.dataset_kwargs["dataset_kwargs_list"],
    #     config.dataset_kwargs["sample_weights"],
    #     config.eval_datasets,
    # )
    # viz_datasets_kwargs_list, _ = filter_eval_datasets(
    #     config.dataset_kwargs["dataset_kwargs_list"],
    #     config.dataset_kwargs["sample_weights"],
    #     config.viz_datasets,
    # )
    # val_callback = ValidationCallback(
    #     loss_fn=loss_fn,
    #     process_batch_fn=lambda batch: process_batch(batch),
    #     text_processor=text_processor,
    #     val_dataset_kwargs_list=val_datasets_kwargs_list,
    #     dataset_kwargs=config.dataset_kwargs,
    #     **config.val_kwargs.to_dict(),
    # )
    # viz_callback = VisualizationCallback(
    #     text_processor=text_processor,
    #     viz_dataset_kwargs_list=viz_datasets_kwargs_list,
    #     dataset_kwargs=config.dataset_kwargs,
    #     **config.viz_kwargs.to_dict(),
    # )
    if "rollout_kwargs" in config:
        rollout_kwargs = config.rollout_kwargs.to_dict()
        dataset_name = rollout_kwargs.pop("dataset_name")
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            action_proprio_metadata=model.dataset_statistics[dataset_name],
            **rollout_kwargs,
        )
    else:
        rollout_callback = None

    wandb_id = "{name}_{time}".format(
        name=FLAGS.name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        id=wandb_id,
        **config.wandb,
    )

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    def eval_viz_rollout(model):
        train_state = TrainState.create(
            rng=jax.random.PRNGKey(1234),
            model=model,
            tx=optax.adamw(optax.constant_schedule(0.0)),  # dummy optimizer
        )

        # # validation metrics
        # val_metrics = val_callback(train_state, step)
        # wandb_log(val_metrics, step=step)

        # # visualizations
        # viz_metrics = viz_callback(train_state, step)
        # wandb_log(viz_metrics, step=step)

        # optional: rollout eval
        if rollout_callback is not None:
            rollout_metrics = rollout_callback(train_state, step)
            wandb_log(rollout_metrics, step=step)

    if FLAGS.step is None:
        list_of_checkpoints = ocp.utils.checkpoint_steps_paths(FLAGS.checkpoints)
        list_of_checkpoints = sorted(
            list_of_checkpoints,
            key=lambda path: ocp.utils.step_from_checkpoint_name(path.name),
        )
        logging.info(list_of_checkpoints)


        for path in list_of_checkpoints:
            step = ocp.utils.step_from_checkpoint_name(path.name)
            if FLAGS.eval_every is not None and step % FLAGS.eval_every != 0:
                continue
            print(f"Loading checkpoint {step}: ", path)
            weights_path, step = download_checkpoint_from_gcs(
                FLAGS.checkpoints,
                step,
                FLAGS.checkpoint_cache_dir,
            )
            params = checkpointer.restore(tf.io.gfile.join(weights_path, step, "default"), model.params)
            model = model.replace(params=params)

            eval_viz_rollout(model)
    else:
        eval_viz_rollout(model)



if __name__ == "__main__":
    app.run(main)
