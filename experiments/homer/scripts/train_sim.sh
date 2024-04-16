export WANDB_MODE=disabled

python scripts/train.py \
    --config experiments/homer/scripts/configs/mujoco_manip.py:vit_s \
    --config.dataset_kwargs.oxe_kwargs.data_dir=/mnt2/homer/datasets/mujoco_rlds \
    --config.dataset_kwargs.batch_size=4 \
    --config.dataset_kwargs.shuffle_buffer_size=100 \
    --config.eval_interval=1 \
    --config.viz_interval=2000000 \
    --name test \
