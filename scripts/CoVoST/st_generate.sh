#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=3

DIRECTION=M2M
COVOST_ROOT=
CHECKPOINT_FILENAME=

for lp in $(ls ${COVOST_ROOT}/test | grep test_st); do
  tgt_lang=$(echo ${lp} | awk -F'_' '{print $4}' | awk -F'.' '{print $1}')
  if [ ${tgt_lang} = "zh-CN" -o ${tgt_lang} = "ja" ]; then
    fairseq-generate ${COVOST_ROOT} \
      --config-yaml config_st.yaml --gen-subset test/${lp} --task speech_to_text \
      --path ${CHECKPOINT_FILENAME} --max-tokens 80000 --beam 5 --prefix-size 1 \
      --scoring sacrebleu --sacrebleu-char-level --skip-invalid-size-inputs-valid-test
  else
    fairseq-generate ${COVOST_ROOT} \
      --config-yaml config_st.yaml --gen-subset test/${lp} --task speech_to_text \
      --path ${CHECKPOINT_FILENAME} --max-tokens 80000 --beam 5 --prefix-size 1 --quiet \
      --scoring sacrebleu --skip-invalid-size-inputs-valid-test
  fi
done
