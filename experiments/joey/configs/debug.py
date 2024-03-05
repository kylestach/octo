import sys

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec
from octo.model.components.action_heads import UNetActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer


def get_config(config_string="full,scripted,no_proprio,64,1"):
    mode, data_type, latefuse_proprio, pred_horizon, act_readout_tokens = config_string.split(",")
    assert mode in ["full", "head_only", "head_mlp_only", "scratch"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    if data_type == "scripted":
        dataset_name = "aloha_sim_cube_scripted_dataset"
    elif data_type == "human":
        dataset_name = "aloha_sim_cube_human_dataset"

    FINETUNING_KWARGS = {
        "name": "bridge_dataset",
        "data_dir": "./tests/debug_dataset",
        "image_obs_keys": {"primary": "image_0", "wrist": None},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "bounds", #"normal", #"bounds",
        # All actions are relative deltas, except for the last one (gripper) which is absolute
        # Specifying this is only necessary if you want to predict > 1 step into the future
        "absolute_action_mask": [False, False, False, False, False, False, True],
        # We also want to avoid normalizing the gripper
        "action_normalization_mask": [True, True, True, True, True, True, False],
        # standardize_fn is dynamically loaded from a file
        # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
        "standardize_fn": ModuleSpec.create(
            "octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform",
        ),
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    }

    if mode in ["full"]:
        frozen_keys = None
        skip_keys = ["heads_*"]
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
        skip_keys = ["heads_*"]
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
        skip_keys = ["heads_*"]
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
        skip_keys = ["heads_*"]
    elif mode == "scratch":
        frozen_keys = None
        skip_keys = ["heads_*", "octo_transformer.*"]
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(100000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path="gs://iliad_europe_west4/octo/checkpoints/octo_base_0127/octo_base",
        pretrained_step=300000,
        batch_size=40, # NOTE: for now traing with smaller batch size
        shuffle_buffer_size=50000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=4,
        save_interval=10000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality="text_conditioned",
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        skip_keys=skip_keys,
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

    # always run ALOHA language conditioned
    goal_relabeling_strategy = "uniform"
    keep_image_prob = 0.0

    pred_horizon = int(pred_horizon)
    act_readout_tokens = int(act_readout_tokens)

    traj_transform_kwargs = dict(
        window_size=window_size,
        future_action_window_size=pred_horizon-1,
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
    # wrist_augment_kwargs = dict(
    #     random_brightness=[0.1],
    #     random_contrast=[0.9, 1.1],
    #     random_saturation=[0.9, 1.1],
    #     random_hue=[0.05],
    #     augment_order=[
    #         "random_brightness",
    #         "random_contrast",
    #         "random_saturation",
    #         "random_hue",
    #     ],
    # )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            # "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs=[
            workspace_augment_kwargs,
            # wrist_augment_kwargs,
        ],
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 32  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs

    # modify the model
    config["config_delete_keys"] = dict(
        model=dict(
            observation_tokenizers=dict(
                wrist=None
            )
        )
    )

    action_head = ModuleSpec.create(
        UNetActionHead,
        readout_key="readout_action",
        use_map=False,
        flatten_tokens=True,
        pred_horizon=pred_horizon,
        action_dim=7,
        n_diffusion_samples=1,
        max_action=1,
        diffusion_steps=100,
    )
    
    if latefuse_proprio == "no_proprio":
        lf_proprio = False
    elif latefuse_proprio == "with_proprio":
        lf_proprio = True
    else:
        raise ValueError

    config["update_config"] = dict(
        model=dict(
            heads=dict(
                action=action_head
            ),
            observation_tokenizers={
                "proprio": ModuleSpec.create(
                    LowdimObsTokenizer,
                    n_bins=-1,
                    obs_keys=["proprio"],
                ),
            },
            latefuse_proprio=lf_proprio,
            readouts={'action': act_readout_tokens}
        )
    )

    # Temp remove rollout vis for debugging.
    # configure the rollout visualizer
    # config["rollout_kwargs"] = dict(
    #     trajs_for_rollouts=5,
    #     visualizer_kwargs_list=[
    #         dict(
    #             env_name="aloha-sim-cube-v0",
    #             max_episode_length=400,
    #             vis_fps=25,
    #             video_subsample_rate=2,
    #             exec_horizon=min(pred_horizon, 25),
    #         ),
    #     ],
    # )

    return ConfigDict(config)
