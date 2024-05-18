export WANDB_MODE=disabled

python scripts/train.py \
    --config experiments/homer/scripts/configs/go1_config.py:vit_t \
    --name test \
