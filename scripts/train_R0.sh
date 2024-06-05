#!/bin/bash

DATASET=$1
NAME=$2

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATASET/bin/ \
    --fp16 \
    --save-dir models/$NAME \
    --tensorboard-logdir tensorboard_logs/$NAME \
    --restore-file models/D0/checkpoint_best.pt \
    --arch bart_large  \
    --task translation  \
    --criterion label_smoothed_cross_entropy  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens 8192  \
    --update-freq 4  \
    --max-epoch 10 \
    --required-batch-size-multiple 1  \
    --seed 42 \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01  \
    --optimizer adam  \
    --adam-betas "(0.9, 0.999)"  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-05  \
    --total-num-update 10000 \
    --warmup-updates 1000  \
    --ddp-backend no_c10d  \
    --num-workers 20 \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --no-epoch-checkpoints \
    --validate-interval-updates 500 \
    --reset-dataloader \
    --reset-meters \
    --reset-optimizer \
    --patience 200 >> models/log/$NAME.log
