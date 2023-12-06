#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=0,2
export LASER=
CoVoST_PATH=
CV13_PATH=

RAW=${CoVoST_PATH}/multi-way-parallel

mkdir -p ${RAW}/{langs,encoding,alignments}

spm_model=${LASER}/nllb/laser2.spm

HIGH-MIDs=(
    ca
    de
    es
    fa
    fr
    it
    nl
    pt
    ru
    zh-CN
)
LOWs=(
    ar
    cy
    et
    id
    ja
    lv
    mn
    sl
    sv-SE
    ta
    tr
)

for lang in ${HIGH-MIDs[@]}; do
    cat ${CoVoST_PATH}/test/test_st_${lang}_en.tsv | awk '{print $4}' > ${RAW}/langs/${lang}.txt
done
for lang in ${LOWs[@]}; do
    cat ${CV13_PATH}/${lang}/test_st_${lang}_en.tsv | awk '{print $4}' > ${RAW}/langs/${lang}.txt
done

for file in $(ls ${RAW}/langs | grep txt | grep en); do
    lang=$(echo $file | awk -F'.' '{print $1}')
    export LANG=${lang}
    echo ${lang}
    test -e ${LASER}/nllb/laser3-${lang}.v1.pt && model_file=${LASER}/nllb/laser3-${lang}.v1.pt || model_file=${LASER}/nllb/laser2.pt

    output_file=${RAW}/encoding/${lang}.enc

    python3 ${LASER}/source/embed.py \
        --input ${RAW}/langs/${file} \
        --encoder ${model_file} \
        --spm-model ${spm_model} \
        --output ${output_file} \
        --verbose
done

LANGs=(
    ar
    ca
    cy
    de
    es
    et
    fa
    fr
    id
    it
    ja
    lv
    mn
    nl
    pt
    ru
    sl
    sv-SE
    ta
    tr
    zh-CN
)

for ((i=0;i<${#LANGs[@]};i++)); do
    for ((j=$(expr ${i} + 1);j<${#LANGs[@]};j++)); do
        l1=${LANGs[i]}
        l2=${LANGs[j]}
        test -e ${RAW}/alignments/${l1}-${l2}.cand-th1.tsv || python3 ${LASER}/source/mine_bitexts.py \
            ${RAW}/langs/${l1}.txt ${RAW}/langs/${l2}.txt \
            --src-lang ${l1} --trg-lang ${l2} \
            --src-embeddings ${RAW}/encoding/${l1}.enc --trg-embeddings ${RAW}/encoding/${l2}.enc \
            --unify --mode mine --retrieval max --margin ratio -k 4  \
            --output ${RAW}/alignments/${l1}-${l2}.cand-th1.tsv --threshold 1.05 \
            --verbose --gpu
    done
done
