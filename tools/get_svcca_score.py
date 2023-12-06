#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cca_core
import json
from tqdm import tqdm

COVOST_PATH=""

def select_singular_values(s, threshold):
    total = np.sum(s)
    s = s / total
    for i in range(len(s)):
        if np.sum(s[:i]) >= threshold:
            return s[:i]
    return s    

def score(lang1, lang2, layer):
    a = np.load(f"{COVOST_PATH}/multi-way-parallel/test/{lang1}_{lang2}/{lang1}_layer_{layer}.npy")
    b = np.load(f"{COVOST_PATH}/multi-way-parallel/test/{lang1}_{lang2}/{lang2}_layer_{layer}.npy")
    ca = (a - np.mean(a, axis=0, keepdims=True)) / np.std(a, axis=0, keepdims=True)
    cb = (b - np.mean(b, axis=0, keepdims=True)) / np.std(b, axis=0, keepdims=True)

    U1, s1, V1 = np.linalg.svd(ca, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cb, full_matrices=False)

    retained_singular_value_s1 = select_singular_values(s1, 0.99)
    adim = len(retained_singular_value_s1)
    retained_singular_value_s2 = select_singular_values(s2, 0.99)
    bdim = len(retained_singular_value_s2)

    sva = np.dot(s1[:adim] * np.eye(adim), V1[:adim])
    svb = np.dot(s2[:bdim] * np.eye(bdim), V2[:bdim])

    results = cca_core.get_cca_similarity(sva, svb, epsilon=1e-10, verbose=False)
    return np.mean(results['cca_coef1'])

def main(layer):
    print("for layer: ", layer)
    Xx2En_LANGs = ["ar", "ca", "cy", "de", "es", "et", "fa", "fr", "id", "it", "ja", "lv", "mn", "nl", "pt", "ru", "sl", "sv-SE", "ta", "tr", "zh-CN"]
    En2Xx_LANGs = ["ar", "ca", "cy", "de", "et", "fa", "id", "ja", "lv", "mn", "sl", "sv-SE", "ta", "tr", "zh-CN"]
    cca_scores = {}
    for l in Xx2En_LANGs:
        cca_scores[l] = {}
    for idx1 in tqdm(range(len(Xx2En_LANGs))):
        for idx2 in range(len(Xx2En_LANGs)):
            if idx2 <= idx1:
                continue
            l1 = Xx2En_LANGs[idx1]
            l2 = Xx2En_LANGs[idx2]

            s = score(l1, l2, layer)
            cca_scores[l1][l2] = s
            cca_scores[l2][l1] = s
    
    with open(f"cca_score_layer_{layer}_Xx2En.json", 'w') as f:
        json.dump(cca_scores, f, indent=4)

if __name__ == "__main__":
    for layer in range(12):
        main(layer)