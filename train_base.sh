accelerate launch train.py \
    --experiment_name 'Conv_ResidualVQVAE_ResNet50Backbone' \
    --working_directory "work_dir" \
    --path_to_data 'data' \
    --checkpoint_dir 'checkpoints' \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_use_rslora \
    --lora_dropout 0.1 \
    --lora_bias 'lora_only' \
    --lora_target_modules 'conv1,conv2,conv3' \
    --lora_exclude_modules 'embedding,upsample,downsample,conv_transpose,conv0' \
    --max_grad_norm 1.0 \
    --per_gpu_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --warmup_epochs 5 \
    --epochs 300 \
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
    --num_workers 8 \
    --base_weights_no_lora 'base_weights' \
    --custom_weight_init \
    --log_wandb
#    --perceptual_loss_lambda 0.1 \
#    --use_perceptual_loss
#    --use_lora
#    --resume_from_checkpoint 'checkpoint_126'
#    --merged_weights_from_lora 'lora_merged_weights/reconstruct.safetensors'
