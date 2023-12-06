#!/bin/bash
set -eo pipefail

DIRECTION=M2M
ROOT=
LANGs=(
    ar
    ca
    cy
    de
    et
    fa
    id
    ja
    lv
    mn
    sl
    sv-SE
    ta
    tr
    zh-CN
)
CHECKPOINT_FILENAME=


for ((i=0;i<${#LANGs[@]};i+=1)); do
    l1=${LANGs[i]}

    export CUDA_VISIBLE_DEVICES=0
    python ./scripts/tools/get_decoder_representations.py ${ROOT} \
      --lang-pair en_${l1} --language ${l1} \
      --config-yaml config_st.yaml --gen-subset test/test_st_en_${l1} --task speech_to_text \
      --path ${CHECKPOINT_FILENAME} --batch-size 64 --beam 5 --prefix-size 1 \
      --scoring sacrebleu --skip-invalid-size-inputs-valid-test

done