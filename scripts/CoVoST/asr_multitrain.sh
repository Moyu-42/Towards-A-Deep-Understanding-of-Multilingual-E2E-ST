#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=0

COVOST_ROOT=
MULTILINGUAL_ASR_SAVE_DIR=

fairseq-train ${COVOST_ROOT} \
    --config-yaml config_asr.yaml \
    --train-subset train_asr \
    --valid-subset dev_asr \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend no_c10d \
    --fp16 \
	--save-dir ${MULTILINGUAL_ASR_SAVE_DIR} --num-workers 1 --max-tokens 160000 --patience 5 \
    --keep-last-epochs 10 \
	--task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
	--arch s2t_transformer_l --ignore-prefix-size 1 --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 2500 --clip-norm 10.0 --seed 3407 --update-freq 8
