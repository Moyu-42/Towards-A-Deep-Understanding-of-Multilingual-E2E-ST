#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os

LANGs = ["ar", "ca", "cy", "de", "es", "et", "fa", "fr", "id", "it", "ja", "lv", "mn", "nl", "pt", "ru", "sl", "sv-SE", "ta", "tr", "zh-CN"]
LR = ["ar", "cy", "et", "id", "ja", "lv", "mn", "nl", "sl", "sv-SE", "ta", "tr"]

DATA_RAW=""
MWP_PATH=""

for lang in LANGs:
    data = pd.read_csv(f"{MWP_PATH}/alignments/{lang}-en.cand-th1.tsv", sep="\t", header=None)
    data.columns = ['score', f'{lang}_text', 'tgt_text']
    # save the file
    data.to_csv(f"{MWP_PATH}/alignments/{lang}-en.th1.tsv", sep="\t", index=False)

# get the original tsv
for idx1 in range(len(LANGs)):
    for idx2 in range(len(LANGs)):
        if idx2 <= idx1:
            continue
        l1 = LANGs[idx1]
        l2 = LANGs[idx2]

        print(f"{l1} -- {l2}")
        try:
            align_df = pd.read_csv(f"{MWP_PATH}/alignments/{l1}-{l2}.cand-th1.tsv", sep='\t', header=None, error_bad_lines=False, nrows=4000)
        except FileNotFoundError:
            continue
        
        if not os.path.exists(f"{MWP_PATH}/test/{l1}_{l2}"): 
            os.makedirs(f"{MWP_PATH}/test/{l1}_{l2}")

        l1_mark = "CoVoST"
        l2_mark = "CoVoST"
        if l1 in LR:
            l1_mark = "CV_13"
        if l2 in LR:
            l2_mark = "CV_13"

        align_df.columns = ["score", f"{l1}_text", f"{l2}_text"]
        align_df = align_df[:3000]
        l1_df_test = pd.read_csv(f"{DATA_RAW}/{l1_mark}/{l1}/test_st_{l1}_en.tsv", sep='\t')
        l2_df_test = pd.read_csv(f"{DATA_RAW}/{l2_mark}/{l2}/test_st_{l2}_en.tsv", sep='\t')
        if l1_mark != "CV_13":
            l1_df_dev = pd.read_csv(f"{DATA_RAW}/{l1_mark}/{l1}/dev_st_{l1}_en.tsv", sep='\t')
            # aggress the test and dev data
            l1_df = l1_df_test.append(l1_df_dev)
        else:
            l1_df = l1_df_test
        if l2_mark != "CV_13":
            l2_df_dev = pd.read_csv(f"{DATA_RAW}/{l2_mark}/{l2}/dev_st_{l2}_en.tsv", sep='\t')
            # aggress the test and dev data
            l2_df = l2_df_test.append(l2_df_dev)
        else:
            l2_df = l2_df_test

        # align_df columns: ['score_x', '{l1}_text', 'tgt_text', 'score_y', '{l2}_text']
        # l1_df columns: ['id', 'audio', "n_frames", "src_text", "tgt_text", "speaker", "src_lang", "tgt_lang"]
        # l2_df columns: ['id', 'audio', "n_frames", "src_text", "tgt_text", "speaker", "src_lang", "tgt_lang"]

        l1_df_new = pd.DataFrame(columns=['id', 'audio', "n_frames", "src_text", "tgt_text", "speaker", "src_lang", "tgt_lang"])
        l2_df_new = pd.DataFrame(columns=['id', 'audio', "n_frames", "src_text", "tgt_text", "speaker", "src_lang", "tgt_lang"])

        # iterate the align_df
        for idx, row in align_df.iterrows():
            l1_text = row[f'{l1}_text']
            l2_text = row[f'{l2}_text']
            l1_df_new = l1_df_new.append(l1_df[l1_df['src_text'] == l1_text])
            l2_df_new = l2_df_new.append(l2_df[l2_df['src_text'] == l2_text])

        if l1 in LR:
            l1_df_new['tgt_text'] = l1_df_new['src_text']
            l1_df_new['src_lang'] = l1
            l1_df_new['tgt_lang'] = 'en'
            l1_df_new = l1_df_new[['id', 'audio', "n_frames", "src_text", "tgt_text", "speaker", "src_lang", "tgt_lang"]]
        if l2 in LR:
            l2_df_new['tgt_text'] = l2_df_new['src_text']
            l2_df_new['src_lang'] = l2
            l2_df_new['tgt_lang'] = 'en'
            l2_df_new = l2_df_new[['id', 'audio', "n_frames", "src_text", "tgt_text", "speaker", "src_lang", "tgt_lang"]]

        # remove the duplicated item based on 'src_text'
        l1_df_new = l1_df_new.drop_duplicates(subset=['src_text'])
        l2_df_new = l2_df_new.drop_duplicates(subset=['src_text'])

        # sort l1_df and l2_df items with the same order as "{MWP_PATH}/bi_mwp/{l1}-{l2}.tsv" column '{l1}_text'
        l1_df = l1_df.sort_values(by=[l1_df['src_text'].isin(align_df[f'{l1}_text'])], ascending=False)
        l2_df = l2_df.sort_values(by=[l2_df['src_text'].isin(align_df[f'{l2}_text'])], ascending=False)

        l1_df_new.to_csv(f"{MWP_PATH}/test/{l1}_{l2}/test_st_{l1}.tsv", sep='\t', index=None)
        l2_df_new.to_csv(f"{MWP_PATH}/test/{l1}_{l2}/test_st_{l2}.tsv", sep='\t', index=None)

