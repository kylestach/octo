from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.data.utils.text_processing import MuseEmbedding
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from octo.model.components.transformer import common_transformer_sizes
from octo.model.components.vit_encoders import ResNet26FILM
from octo.utils.spec import ModuleSpec

import octo, os, sys, functools
sys.path.append(os.path.join(os.path.dirname(octo.__file__), "../"))
from experiments.sudeep.aloha.resnet_pt import resnet_26_loader

# some constants for aloha
ADIM = 14
IMG_SIZE = (224, 224)


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
def get_config(
    H=100,
    task_cond="lang",
    model="detr",
):
    print("Creating config with: ", locals())
    action_horizon = int(H)
    num_steps = FieldReference(default=150000)
    window_size = FieldReference(default=1)

    return ConfigDict(
        dict(
            seed=42,
            num_steps=num_steps,
            save_dir='gs://multi-robot-bucket2/runs/',
            model=get_model_config(model, action_horizon),
            window_size=window_size,
            dataset_kwargs=get_dataset_config(task_cond, window_size, action_horizon),
            optimizer=dict(
                learning_rate=dict(
                    name="cosine",
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=5000,
                    decay_steps=num_steps,
                    end_value=1e-8,
                ),
                weight_decay=1.0e-6,
                clip_gradient=1.0,
                b1=0.95,
                b2=0.999,
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=placeholder(int),
            log_interval=500,

            eval_interval=10000,
            viz_interval=5e20, # TODO fix vis for aloha

            save_interval=25000,
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
            text_processor=ModuleSpec.create(MuseEmbedding),
            pretrained_loaders=(ModuleSpec.create(resnet_26_loader, restore_path='gs://sudeep_r2d2_experiments/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz'),),
            wandb=dict(
                project="octo",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            wandb_resume_id=placeholder(str),
            eval_datasets=(
                "aloha_pen_uncap_diverse_dataset",
            ),
        )
    )


def get_dataset_config(task_cond, window_size, action_horizon):
    traj_transform_kwargs, frame_transform_kwargs = get_augmentation_config(task_cond, window_size, action_horizon)

    base_aloha_kwargs = dict(
                        data_dir="gs://rail-orca-central2/",
                        image_obs_keys= {"high": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"},
                        proprio_obs_key="state",
                        action_normalization_mask=[True] * ADIM,
                        language_key= "language_instruction",
                        skip_proprio_norm=True,
                    )
    return dict(
            dataset_kwargs_list=[
                        dict(
                        name="aloha_pen_uncap_diverse_dataset",
                        **base_aloha_kwargs),],
            sample_weights=[1],
            traj_transform_kwargs=traj_transform_kwargs,
            frame_transform_kwargs=frame_transform_kwargs,
            batch_size=512,
            shuffle_buffer_size=100000,
            balance_weights=True,
            traj_transform_threads=64,  # shared between all datasets
            traj_read_threads=64,  # shared between all datasets
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
        max_action_dim=ADIM,
        goal_relabeling_strategy="uniform",
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )

    image_augment_kwargs = dict(
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
            "high": IMG_SIZE,         # top down workspace camera
            "left_wrist": IMG_SIZE,   # left wrist camera
            "right_wrist": IMG_SIZE,  # right wrist camera
        },
        image_augment_kwargs={
            "high": image_augment_kwargs,
            "left_wrist": image_augment_kwargs,
            "right_wrist": image_augment_kwargs
        },
    )
    return traj_transform_kwargs, frame_transform_kwargs


def get_model_config(transformer_size, action_horizon):
    """
    This model stacks all the images from different cameras together, and passes it through
    a small convolutional stem before entering the transformer.

    The action head pools all the observation token embeddings, and passes it through a small MLP
    before predicting the action using a MSE loss.
    """
    token_embedding_size, transformer_kwargs = common_transformer_sizes(
        transformer_size
    )

    encoder = ModuleSpec.create(ResNet26FILM, use_film=True)
    return dict(
        observation_tokenizers=dict(
            high=ModuleSpec.create(
               ImageTokenizer,
               obs_stack_keys=["image_high"],
               task_stack_keys=["image_high"],
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
           proprio=ModuleSpec.create(
             LowdimObsTokenizer,
             obs_keys=["proprio"],
             dropout_rate=0.2,
           ),
        ),
        task_tokenizers=dict(),
        heads=dict(
            action=ModuleSpec.create(
                L1ActionHead,
                action_horizon=action_horizon,
                action_dim=ADIM,
                num_preds=ADIM,
                pool_strategy="pass",
                readout_key="readout_action",
                clip_pred=False,
            ),
        ),
        use_correct_attention=True,
        repeat_task_tokens=True,
        readouts=dict(action=action_horizon),
        token_embedding_size=token_embedding_size,
        transformer_kwargs=transformer_kwargs,
        max_horizon=action_horizon,
    )
