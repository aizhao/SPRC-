#!/bin/bash

# 方案1: 只使用在线难负样本挖掘（推荐先用这个）
echo "=== Training with online hard negative mining ==="
CUDA_VISIBLE_DEVICES=0 python src/blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 10 \
    --batch-size 64 \
    --learning-rate 5e-6 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    --target-ratio 1.25 \
    --transform targetpad \
    --use-region-loss \
    --box-file roi_align_integration/data/cirr_boxes_train.json \
    --use-hard-negatives \
    --hard-negative-ratio 0.5 \
    --hard-negative-weight 0.5

# 方案2: 使用预计算的难负样本数据库（需要先运行预处理）
# echo "=== Training with pre-computed hard negatives ==="
# CUDA_VISIBLE_DEVICES=0 python src/blip_fine_tune_2.py \
#     --dataset CIRR \
#     --blip-model-name blip2_cir_align_prompt \
#     --backbone pretrain \
#     --num-epochs 10 \
#     --batch-size 64 \
#     --learning-rate 5e-6 \
#     --loss-align 0.4 \
#     --loss-rtc 0.4 \
#     --validation-frequency 1 \
#     --save-training \
#     --save-best \
#     --target-ratio 1.25 \
#     --transform targetpad \
#     --use-region-loss \
#     --box-file roi_align_integration/data/cirr_boxes_train.json \
#     --use-hard-negatives \
#     --hard-negative-ratio 0.5 \
#     --hard-negative-weight 0.5 \
#     --hard-negative-db hard_negatives/cirr_hard_negatives.json
