#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=0,1

DIRECTION=M2M # O2M M2O and M2M
COVOST_ROOT=
MULTILINGUAL_ST_SAVE_DIR=
PRETRAINED_PATH=

M2O_VALID_SUBSET=dev/dev_st_ar_en,dev/dev_st_ca_en,dev/dev_st_cy_en,dev/dev_st_de_en,dev/dev_st_es_en,dev/dev_st_et_en,dev/dev_st_fa_en,dev/dev_st_fr_en,dev/dev_st_id_en,dev/dev_st_it_en,dev/dev_st_ja_en,dev/dev_st_lv_en,dev/dev_st_mn_en,dev/dev_st_nl_en,dev/dev_st_pt_en,dev/dev_st_ru_en,dev/dev_st_sl_en,dev/dev_st_sv-SE_en,dev/dev_st_ta_en,dev/dev_st_tr_en,dev/dev_st_zh-CN_en
M2O_TRAIN_SUBSET=train/train_st_ar_en,train/train_st_ca_en,train/train_st_cy_en,train/train_st_de_en,train/train_st_es_en,train/train_st_et_en,train/train_st_fa_en,train/train_st_fr_en,train/train_st_id_en,train/train_st_it_en,train/train_st_ja_en,train/train_st_lv_en,train/train_st_mn_en,train/train_st_nl_en,train/train_st_pt_en,train/train_st_ru_en,train/train_st_sl_en,train/train_st_sv-SE_en,train/train_st_ta_en,train/train_st_tr_en,train/train_st_zh-CN_en

O2M_VALID_SUBSET=dev/dev_st_en_ar,dev/dev_st_en_ca,dev/dev_st_en_cy,dev/dev_st_en_de,dev/dev_st_en_et,dev/dev_st_en_fa,dev/dev_st_en_id,dev/dev_st_en_ja,dev/dev_st_en_lv,dev/dev_st_en_mn,dev/dev_st_en_sl,dev/dev_st_en_sv-SE,dev/dev_st_en_ta,dev/dev_st_en_tr,dev/dev_st_en_zh-CN,dev_st
O2M_TRAIN_SUBSET=train_st

M2M_VALID_SUBSET=${M2O_VALID_SUBSET},${O2M_VALID_SUBSET}
M2M_TRAIN_SUBSET=train/train_st_ar_en,train/train_st_ca_en,train/train_st_cy_en,train/train_st_de_en,train/train_st_en_ar,train/train_st_en_ca,train/train_st_en_cy,train/train_st_en_de,train/train_st_en_et,train/train_st_en_fa,train/train_st_en_id,train/train_st_en_ja,train/train_st_en_lv,train/train_st_en_mn,train/train_st_en_sl,train/train_st_en_sv-SE,train/train_st_en_ta,train/train_st_en_tr,train/train_st_en_zh-CN,train/train_st_es_en,train/train_st_et_en,train/train_st_fa_en,train/train_st_fr_en,train/train_st_id_en,train/train_st_it_en,train/train_st_ja_en,train/train_st_lv_en,train/train_st_mn_en,train/train_st_nl_en,train/train_st_pt_en,train/train_st_ru_en,train/train_st_sl_en,train/train_st_sv-SE_en,train/train_st_ta_en,train/train_st_tr_en,train/train_st_zh-CN_en

fairseq-train ${COVOST_ROOT} \
    --config-yaml config_st.yaml \
    --train-subset ${M2M_TRAIN_SUBSET}\
    --valid-subset ${M2M_VALID_SUBSET} \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend no_c10d \
    --fp16 \
	--save-dir ${MULTILINGUAL_ST_SAVE_DIR} --num-workers 1 --max-tokens 160000 --patience 5 \
    --log-interval 100 \
	--task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
	--arch s2t_transformer_l --ignore-prefix-size 1 --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 2500 --clip-norm 10.0 --seed 3407 --update-freq 12 \
    --load-pretrained-encoder-from ${PRETRAINED_PATH}
