from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.data.utils.text_processing import UniversalSentenceEncoder
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from octo.model.components.transformer import common_transformer_sizes
from octo.model.components.vit_encoders import ResNet26, ResNet26FILM
from octo.utils.spec import ModuleSpec

import octo, os, sys, functools

sys.path.append(os.path.join(os.path.dirname(octo.__file__), "../"))
from experiments.sudeep.aloha.resnet_pt import resnet_26_loader

BIMANUAL_ACTION_DIM = 14
SINGLE_ARM_ACTION_DIM = 7
NAV_ACTION_DIM = 2
QUADRUPED_ACTION_DIM = 12

HEAD_TO_DATASET = {
    "single_arm": [
        "droid_wipe",
    ],
}


def wrap(f):
    """Simple wrapper to enable passing config strings to `get_config`

    Usage:

    python train.py --config=config.py:vit_s,multimodal
    python train.py --config=config.py:transformer_size=vit_s
    """

    @functools.wraps(f)
    def wrapped_f(config_string=None):
        if config_string is None:
            return f()
        elements = config_string.split(",")
        args, kwargs = [], {}
        for e in elements:
            if "=" in e:
                k, v = e.split("=")
                kwargs[k] = v
            else:
                args.append(e)
        return f(*args, **kwargs)

    return wrapped_f


@wrap
def get_config():
    print("Creating config with: ", locals())
    window_size = FieldReference(default=5)

    return ConfigDict(
        dict(
            seed=42,
            num_steps=300000,
            save_dir="gs://multi-robot-bucket2/runs",
            model=get_model_config("vit_t_repeat"),
            window_size=window_size,
            dataset_kwargs=get_dataset_config("multi", window_size, 100),
            optimizer=dict(
                learning_rate=dict(
                    name="rsqrt",
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=2000,
                    timescale=10000,
                ),
                weight_decay=0.1,
                clip_gradient=1.0,
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=placeholder(int),
            log_interval=500,
            eval_interval=20000,
            viz_interval=5e20,
            save_interval=10000,
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
            resume_path=placeholder(str),
            text_processor=ModuleSpec.create(UniversalSentenceEncoder),
            pretrained_loaders=(
                ModuleSpec.create(
                    resnet_26_loader,
                    restore_path="gs://sudeep_r2d2_experiments/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
                ),
            ),
            wandb=dict(
                project="octo",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            wandb_resume_id=placeholder(str),
            eval_datasets=(
                "droid_wipe",
            ),
            viz_datasets=()
        )
    )


def get_dataset_config(task_cond, window_size, action_horizon):
    traj_transform_kwargs, frame_transform_kwargs = get_augmentation_config(
        task_cond, window_size, action_horizon
    )

    return dict(
        oxe_kwargs=dict(
            data_mix=[("droid_wipe", 1.0)],
            data_dir="gs://rail-orca-central2/resize_256_256/",
            load_camera_views=("primary","left_wrist", "right_wrist",),
            load_proprio=True,
            load_depth=False,
        ),
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        batch_size=64,
        shuffle_buffer_size=50000,
        balance_weights=False,
        traj_transform_threads=48,
        traj_read_threads=48,
    )


def get_augmentation_config(task_cond, window_size, action_horizon):
    if task_cond == "image":
        keep_image_prob = 1.0
    elif task_cond == "lang":
        keep_image_prob = 0.0
    elif task_cond == "multi":
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=action_horizon,
        max_action_dim=SINGLE_ARM_ACTION_DIM,
        head_to_dataset=HEAD_TO_DATASET,
        goal_relabeling_strategy="uniform",
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # TODO: fine to not have this for aloha and bridge, should check that we don't need to subsample for other datasets
        # subsample_length=100,
    )

    bridge_image_augment_kwargs = dict(
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

    aloha_image_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.9, 1.0], ratio=[0.75, 4.0 / 3]),
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

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (224, 224),
            "left_wrist": (224, 224),
            "right_wrist": (224, 224),
        },
        image_augment_kwargs={
            "primary": bridge_image_augment_kwargs,
            "left_wrist": aloha_image_augment_kwargs,
            "right_wrist": aloha_image_augment_kwargs,
        },
        num_parallel_calls=200,
    )
    return traj_transform_kwargs, frame_transform_kwargs


def get_model_config(transformer_size):
    """
    This model stacks all the images from different cameras together, and passes it through
    a small convolutional stem before entering the transformer.

    The action head pools all the observation token embeddings, and passes it through a small MLP
    before predicting the action using a MSE loss.
    """
    token_embedding_size, transformer_kwargs = common_transformer_sizes(
        transformer_size
    )

    encoder = ModuleSpec.create(ResNet26FILM)
    return dict(
        observation_tokenizers=dict(
            primary=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_primary"],
                task_stack_keys=["image_primary"],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            left=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_left_wrist"],
                task_stack_keys=[],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            right=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_right_wrist"],
                task_stack_keys=[],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
        ),
        task_tokenizers=dict(),
        heads=dict(
            single_arm=ModuleSpec.create(
                L1ActionHead,
                action_horizon=4,
                action_dim=SINGLE_ARM_ACTION_DIM,
                num_preds=SINGLE_ARM_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_single_arm",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True
            ),

        ),
        use_correct_attention=True,
        repeat_task_tokens=True,
        readouts=dict(
            single_arm=4,
        ),
        token_embedding_size=token_embedding_size,
        transformer_kwargs=transformer_kwargs,
        max_horizon=10,
    )
