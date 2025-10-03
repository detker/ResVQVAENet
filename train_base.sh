#!/bin/bash

accelerate launch train.py \
    --experiment_name 'Conv_ResidualVQVAE_ResNet50Backbone' \
    --working_directory "work_dir" \
    --path_to_data 'data' \
    --checkpoint_dir 'checkpoints' \
    --max_grad_norm 1.0 \
    --per_gpu_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --warmup_epochs 3 \
    --epochs 200 \
    --save_checkpoint_interval 1 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --commitment_loss_beta 0.25 \
    --ema_decay 0.999 \
    --max_no_of_checkpoints 5 \
    --img_size 224 \
    --in_channels 3 \
    --num_workers 5 \
    --custom_weight_init \
    --perceptual_loss_lambda 1e-3 \
    --use_perceptual_loss \
    --log_wandb
#    --resume_from_checkpoint 'checkpoint_8'

