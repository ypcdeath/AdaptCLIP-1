#!/bin/bash

cd /root/autodl-tmp/AdaptCLIP
device=0
data_root=/root/autodl-tmp/datasets
test_dataset=mvtec
n_ctx=12
vl_reduction=4
pq_mid_dim=128
train_dataset=mvtec
base_dir=${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_${train_dataset}_3adapters_batch8
model_dir=./adaptclip_checkpoints/${base_dir}
test_data_path=/root/autodl-tmp/datasets/MVTec


for SHOTS in 0 1 2 4 8 16; do

    echo "=== Starting AdaptCLIP Few-Shot Testing: ${SHOTS}-shot ==="


    save_dir=./results/MVTec_fixed_fewshot/${SHOTS}shot


    mkdir -p ${save_dir}


    CUDA_VISIBLE_DEVICES=${device} python test.py \
        --dataset ${test_dataset} \
        --test_data_path ${test_data_path} \
        --seed 2026 \
        --k_shots ${SHOTS} \
        --checkpoint_path ${model_dir}/epoch_15.pth \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 \
        --image_size 518 \
        --batch_size 8 \
        --n_ctx ${n_ctx} \
        --vl_reduction ${vl_reduction} \
        --pq_mid_dim ${pq_mid_dim} \
        --visual_learner \
        --textual_learner \
        --pq_learner \
        --pq_context \
        > ${save_dir}/few_shot_${SHOTS}.log 2>&1


    echo "=== Finished ${SHOTS}-shot ==="

done