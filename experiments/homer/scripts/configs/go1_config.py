from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict, FieldReference

get_base_config = imp.load_source("config", "scripts/configs/config.py").get_config

from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.utils.spec import ModuleSpec
from octo.data.utils.text_processing import HFTokenizer
import experiments.homer.sim.wrappers.go1_wrapper

HEAD_TO_DATASET = {"quadruped": ["go1"]}


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    config = get_base_config(config_string)

    action_horizon = 1
    window_size = 5
    action_dim = FieldReference(12)

    config["model"]["observation_tokenizers"] = {
        "quadruped": ModuleSpec.create(
            LowdimObsTokenizer,
            obs_keys=["proprio_quadruped"],
        ),
    }

    config["model"]["readouts"] = {"action": 1}
    del config["model"]["heads"]["action"]
    config["model"]["heads"]["quadruped"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=action_horizon,
        action_dim=action_dim,
        pool_strategy="use_map",
        readout_key="readout_action",
        clip_pred=False,
        loss_weight=1.0,
        constrain_loss_dims=True,
    )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]
    del config["text_processor"]
    del config["pretrained_loaders"]
    del config["dataset_kwargs"]["traj_transform_kwargs"]["task_augment_kwargs"]
    del config["dataset_kwargs"]["oxe_kwargs"]["data_mix"]

    config["val_kwargs"]["modes_to_evaluate"] = ("text_conditioned",)

    base_rollout_vis_kwargs = dict(
        env_name="go1",
        max_episode_length=300,
        exec_horizon=action_horizon,
        history_length=window_size,
        vis_fps=10,
        video_subsample_rate=5,
        use_temp_ensembling=False,
        env_kwargs=dict(),
        video_obs_key="image_video",
        head_name="quadruped",
    )

    config = update_config(
        config,
        num_steps=300000,
        window_size=window_size,
        eval_interval=1000,
        viz_interval=1000,
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix=[("go1", 1.0)],
                data_dir="/mnt2/homer/datasets/go1_rlds",
                load_camera_views=(),
                load_depth=False,
                load_proprio=True,
                force_recompute_dataset_statistics=False,
            ),
            traj_transform_kwargs=dict(
                head_to_dataset=HEAD_TO_DATASET,
                action_horizon=action_horizon,
                max_action_dim=action_dim,
                task_augment_strategy="delete_task_conditioning",
                task_augment_kwargs=dict(
                    keep_image_prob=0.0,
                ),
            ),
            batch_size=256,
            shuffle_buffer_size=500000,
            balance_weights=True,
        ),
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(),
        eval_datasets=["go1"],
        viz_datasets=[],
        rollout_kwargs=dict(
            dataset_name="go1",
            modes_to_evaluate=("text_conditioned",),
            trajs_for_rollouts=10,
            visualizer_kwargs_list=[
                dict(
                    **base_rollout_vis_kwargs,
                    name="go1",
                )
            ],
        ),
    )

    return config
