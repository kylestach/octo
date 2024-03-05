from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec
from octo.model.components.action_heads import UNetActionHead

import octo, os, sys
sys.path.append(os.path.join(os.path.dirname(octo.__file__), "../"))

def get_config(config_string="4,rel_r6,wrist"):
    pred_horizon, act_type, cams = config_string.split(',')

    # hard-code some constants to match the original configs
    task = 'language_conditioned'
    mode = 'full'

    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]
    assert cams in ["wrist", "agent"]
    assert act_type in ["rel", "abs", "rel_r6"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    if act_type == "abs":
        n_act_dims = 10
        standardize_fn = "experiments.joey.transforms:iliad_franka_dataset_transform_abs"
    elif act_type == "rel_r6":
        n_act_dims = 10
        standardize_fn = "experiments.joey.transforms:iliad_franka_dataset_transform_rel_r6"
    elif act_type == "rel":
        n_act_dims = 7
        standardize_fn = "experiments.joey.transforms:iliad_franka_dataset_transform_rel"
    else:
        raise ValueError("Incorrect act_type in config string.")

    if cams == "wrist":
        image_obs_keys = {"primary": "agent_image", "wrist": "wrist_image"}
    else:
        image_obs_keys = {"primary": "agent_image", "wrist": None}

    FINETUNING_KWARGS = {
        "name": "franka_coffee_pod_3",
        "data_dir": "gs://iliad_europe_west4/octo/finetuning_datasets",
        "image_obs_keys": image_obs_keys,
        "proprio_obs_key": "state",
        "language_key": "language_instruction",
        # "state_encoding": StateEncoding.POS_QUAT,
        # "action_encoding": ActionEncoding.EEF_POS,
        "action_normalization_mask": [True] * (n_act_dims-1) + [False],
        "standardize_fn": ModuleSpec.create(standardize_fn),
        "num_parallel_reads": 6,  # for reading from disk / GCS OVERRIDE: Set to 2 instead of 8
        "num_parallel_calls": 12,  # for initial dataset construction
        "force_recompute_dataset_statistics": True,
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(500000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=256,
        shuffle_buffer_size=40000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1e9,  # disable eval due to bugs
        save_interval=50000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict( # copied from diffusion policy
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=500,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=1e-6,
            b1=0.95,
            b2=0.999,
            eps=1e-8,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    pred_horizon = int(pred_horizon)

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=pred_horizon,
        max_action_dim=n_act_dims,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
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
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs={
            "primary": workspace_augment_kwargs,
            "wrist": wrist_augment_kwargs,
        },
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs

    action_head = ModuleSpec.create(
        UNetActionHead,
        action_dim=n_act_dims,
        readout_key="readout_action",
        n_diffusion_samples=1,
        use_map=False,
        flatten_tokens=False,
        max_action=1,
        diffusion_steps=100,
        down_dims=[128, 256, 512]
    )
    config["update_config"] = dict(
        model=dict(
            heads=dict(
                action=action_head
            ),
        )
    )
    return ConfigDict(config)
