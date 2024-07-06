#!/bin/bash
python train_GTSA_cls.py \
    --num_workers 6 \
    --seed 1024 \
    --use_avg_pool False \
    --src_dataset 'modelnet' \
    --trgt_dataset 'shapenet' \
    --epochs 100 \
    --gpus '0' \
    --batch_size 8 \
    --test_batch_size 8 \
    --use_aug False \
    --lambda_0 0.25 \
    --epoch_warmup 10 \
    --selection_strategy 'ratio' \
    --use_gradual_src_threshold False \
    --use_gradual_trgt_threshold True \
    --mode_src_threshold 'nonlinear' \
    --mode_trgt_threshold 'nonlinear' \
    --exp_k 0.15 \
    --src_threshold 0.0 \
    --trgt_threshold 1.0 \
    --use_gradual_src_ratio True \
    --use_gradual_trgt_ratio True \
    --src_ratio 1.0 \
    --trgt_ratio 1.0 \
    --period_update_pool 10 \
    --use_model_eval True \
    --loss_function 'use_mean' \
    --use_EMA False \
    --EMA_update_warmup False \
    --EMA_decay 0.99 \
    --use_src_IDFA True \
    --use_trgt_IDFA True \
    --tao 2.0 \
    --w_PCFEA 1.0 \
    --w_src_IDFA 1.0 \
    --w_trgt_IDFA 1.0 \
    --exp_name 'test'