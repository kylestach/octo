from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
).get_config

from octo.model.components.action_heads import DiffusionActionHead, MSEActionHead
from octo.model.components.tokenizers import ImageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from typing import Dict, Any
import tensorflow as tf

def len_greater_than_one(trajectory: Dict[str, Any]):
    return tf.shape(trajectory["action"])[0] > 1

def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    config = get_base_config(config_string)

    action_dim = 2

    config["window_size"] = 2
    config["num_steps"] = 300000
    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
    }
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["heads"]["action"] = ModuleSpec.create(
        MSEActionHead,
        readout_key="readout_action",
        use_map=False,
        action_horizon=4, # changed from 4 to 1 after removing transform
        action_dim=action_dim,
    )

    # We augment differently for the primary and wrist cameras
    primary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]


    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
    }

    del config["dataset_kwargs"]["oxe_kwargs"]["data_mix"]
    del config["pretrained_loaders"]

    config["dataset_kwargs"]["traj_transform_kwargs"]["task_augment_strategy"] = None
    del config["dataset_kwargs"]["traj_transform_kwargs"]["task_augment_kwargs"]

    config = update_config(
        config,
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="gnm_only_mix",
                data_dir="gs://gnm_rlds_separate",
                load_camera_views=("primary",),
                load_depth=False,
                force_recompute_dataset_statistics=True, # recompute dataset stats now that we're removing transform
                filter_functions=[ModuleSpec.create(len_greater_than_one)],
                # skip_norm=True,
            ),
            traj_transform_kwargs=dict(
                action_horizon=4, # change action horizon to 1 when removing transform
                max_action_dim=action_dim,
                goal_relabeling_kwargs=dict(
                    max_goal_distance=15,
                )
            ),
            batch_size=256,
            shuffle_buffer_size=500000,
            balance_weights=True,
        ),

        pretrained_loaders=(),
        # eval_datasets=['sacson_dataset'],
        eval_datasets=["cory_hall_dataset", "go_stanford_dataset", "recon_dataset", "sacson_dataset", "scand_dataset", "seattle_dataset", "tartan_drive_dataset"],
        log_interval=100,
        eval_interval=5000,
        viz_interval=2000000,
        save_interval=10000,
    )

    return config
