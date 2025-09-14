#!/bin/bash

python train.py \
  --_model_weights_dir 'weights'\
  --dataset MNIST \
  --img_wh 32 \
  --latent_space_dim 4 \
  --in_channels 1 \
  --batch_size 64 \
  --iterations 25000 \
  --eval_intervals 250 \
  --num_workers 8 \
  --kl_divergence_weight 1.0 \
  --lr 0.0005 \
  --_variational
