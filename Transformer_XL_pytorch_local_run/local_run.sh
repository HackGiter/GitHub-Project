#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ./data/text8/ \
        --dataset text8 \
        --n_layer 4 \
        --d_model 128 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 512 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 400 \
        --tgt_len 128 \
        --mem_len 128 \
        --eval_tgt_len 64 \
        --batch_size 22 \
        --multi_gpu \
        --gpu0_bsz 1 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
