#!/bin/bash
set -eo pipefail

DIRECTION=M2M
COVOST_PATH=
LANGs=$(ls ${ROOT}/test | grep "test_st" | grep "_en.tsv")
CHECKPOINT_FILENAME=

for file in ${LANGs}; do
    lp=$(echo ${lp} | cut -d '_' -f 3- | awk -F'.' '{print $1}')
    l1=$(echo $lp | cut -d '_' -f 1)
    l2=$(echo $lp | cut -d '_' -f 2)
    echo "generating ${lp}"

    export CUDA_VISIBLE_DEVICES=1
    python ./scripts/tools/get_representations.py ${ROOT} \
      --lang-pair ${lp} --language ${l1} \
      --config-yaml config_st.yaml --gen-subset test/test_st_${l1} --task speech_to_text \
      --path ${CHECKPOINT_FILENAME} --batch-size 64 --beam 5 --prefix-size 1 \
      --scoring sacrebleu --skip-invalid-size-inputs-valid-test

    export CUDA_VISIBLE_DEVICES=1
    python ./scripts/tools/get_representations.py ${ROOT} \
      --lang-pair ${lp} --language ${l2} \
      --config-yaml config_st.yaml --gen-subset test/test_st_${l2} --task speech_to_text \
      --path ${CHECKPOINT_FILENAME} --batch-size 512 --beam 5 --prefix-size 1 \
      --scoring sacrebleu --skip-invalid-size-inputs-valid-test
done
