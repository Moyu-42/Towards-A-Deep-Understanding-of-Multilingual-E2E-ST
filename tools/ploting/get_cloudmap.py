#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LANGs = ["ar", "ca", "cy", "de", "et", "fa", "id", "ja", "lv", "mn", "sl", "sv-SE", "ta", "tr", "zh-CN"]
HIGHs = ["fr", "de", "ca", "es"]
MIDs = ["fa", "it", "ru", "pt", "zh-CN", "nl"]
LOWs = ["ar", "cy", "et", "id", "ja", "lv", "mn", "sl", "sv-SE", "ta", "tr"]
LANGs = HIGHs + MIDs + LOWs

COVOST_PATH=""

def get_mwp_lang(lang, layer, n_per_lang=300):
    res = []
    for lang_ in LANGs:
        if lang < lang_:
            lang_rep = np.load(f"{COVOST_PATH}/multi-way-parallel/test/{lang}_{lang_}/{lang}_layer_{layer}.npy", allow_pickle=False)
            lang_rep = lang_rep[:n_per_lang]
        elif lang > lang_:
            lang_rep = np.load(f"{COVOST_PATH}/multi-way-parallel/test/{lang_}_{lang}/{lang}_layer_{layer}.npy", allow_pickle=False)
            lang_rep = lang_rep[:n_per_lang]
        else:
            continue
        res.append(lang_rep)
    res = np.concatenate(res, axis=0)
    return res

def get_lang_lda(subset, langs, layer, n_per_lang=300):
    if os.path.exists(f"{COVOST_PATH}/multi-way-parallel/{subset}/lang_lda_layer{layer}.npy"):
        return np.load(f"{COVOST_PATH}/multi-way-parallel/{subset}/lang_lda_layer{layer}.npy", allow_pickle=False)
    else:
        all_lang_reps = []
        all_lang_labels = []
        for lang_i, lang in enumerate(langs):
            if subset != "test":
                lang_reps = np.load(f"{COVOST_PATH}/multi-way-parallel/{subset}/{lang}_layer_{layer}.npy", allow_pickle=False)
                lang_reps = lang_reps - np.mean(lang_reps, axis=0, keepdims=True)
                if n_per_lang < lang_reps.shape[0]:
                    to_keep = np.random.choice(lang_reps.shape[0], size=n_per_lang, replace=False)
                    lang_reps = lang_reps[to_keep]
            else:
                lang_reps = get_mwp_lang(lang, layer, n_per_lang)
            all_lang_reps.append(lang_reps)
            all_lang_labels.append(np.ones(lang_reps.shape[0], dtype=np.int32) * lang_i)
        all_lang_reps = np.concatenate(all_lang_reps, axis=0) # Concatenate across languages.
        all_lang_labels = np.concatenate(all_lang_labels, axis=0)
        lda = LinearDiscriminantAnalysis()
        lda.fit(all_lang_reps, all_lang_labels)
        # Shape: (n_dims, n_classes-1)
        lda_axes = lda.scalings_
        np.save(f"{COVOST_PATH}/multi-way-parallel/{subset}/lang_lda_layer{layer}.npy", lda_axes, allow_pickle=False)
    return lda_axes

def get_direction(subset, axis_code, layer):
    lang_lda = get_lang_lda(subset, LANGs, layer)
    return lang_lda[:, axis_code]

def get_orthonormal(vector, axis_directions):
    proj_matrix = np.linalg.inv(np.matmul(axis_directions.T, axis_directions))
    proj_matrix = np.matmul(np.matmul(axis_directions, proj_matrix), axis_directions.T)
    projected_component = np.matmul(proj_matrix, vector)
    orthogonal_component = vector - projected_component
    return orthogonal_component / np.linalg.norm(orthogonal_component)

def get_reps_with_token_info(subset, lang, layer, n_reps, max_seq_length=512):
    # Load representations.
    if subset != "test":
        reps = np.load(f"{COVOST_PATH}/multi-way-parallel/{subset}/{lang}_layer_{layer}.npy", allow_pickle=False)
    else:
        reps = get_mwp_lang(lang, layer)
    to_keep = np.random.choice(reps.shape[0], size=n_reps, replace=False)
    to_keep_bool = np.zeros(reps.shape[0], dtype=bool)
    to_keep_bool[to_keep] = True
    reps_to_return = reps[to_keep_bool].copy()
    return reps_to_return

def project_reps(axis_directions, points, subspace_m):
    subspace_k = axis_directions.shape[1]
    inner_products = np.dot(axis_directions.T, axis_directions) # k x k
    if not np.allclose(inner_products, np.identity(subspace_k)):
        print("WARNING: basis not orthonormal.")
    projected_points = np.matmul(axis_directions.T, (points - subspace_m.reshape(1, -1)).T).T
    return projected_points

def plot_3d(layer, subset):
    axis_origin = np.zeros(1)
    axis_directions = []
    axis_directions.append(get_direction(subset, 0, layer))
    axis_directions.append(get_direction(subset, 1, layer))
    axis_directions.append(get_direction(subset, 2, layer))
    axis_directions = np.stack(axis_directions, axis=-1)
    axis_directions[:, 0] = axis_directions[:, 0] / np.linalg.norm(axis_directions[:, 0])
    axis_directions[:, 1] = get_orthonormal(axis_directions[:, 1], axis_directions[:, [0]])
    axis_directions[:, 2] = get_orthonormal(axis_directions[:, 2], axis_directions[:, :2])
    print("Getting points.")
    plot_langs = LANGs
    reps_dict = dict() # Map languages to representations.
    token_info_dict = dict() # Map languages to token info.
    for lang in plot_langs:
        print("Getting points for lang: {}".format(lang))
        reps = get_reps_with_token_info(subset, lang, layer, 128)
        reps_dict[lang] = reps

    print("Plotting.")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')

    # generate 21 different colors, one for each language
    # the first 4 colors need to be in red colors, the 5th-11th colors need to be in bleu colors, the rest colors need to be in green colors
    colors = ["#FF0000", "#FF3300", "#FF8080", "#FF6666", "#99CCFF", "#0000FF", "#3333FF", "#8080FF", "#6699FF", "#0000CC",  "#FFFF00", "#FFFF80", "#FFCC00", "#FFFF33", "#FFCC66", "#FFCC33", "#FFCC99", "#FFFF66", "#FFCC00", "#FF9933", "#FFCC33"]

    for i, (label, reps) in enumerate(reps_dict.items()):
        projected_points = project_reps(axis_directions, reps, axis_origin)
        marker = 'o'
        if len(LANGs) == 21:
            if LANGs[i] in HIGHs:
                marker = '.'
            if LANGs[i] in LOWs:
                marker = '2'
            if LANGs[i] in MIDs:
                marker = '+'
        ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], label=label, c=colors[i], marker=marker, alpha=1.0, s=15)
    legend = plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    for lh in legend.legendHandles:
        lh.set_alpha(1.0)
        lh.set_sizes([50.0])

    ax.view_init(elev=30, azim=70)
    plt.title(f"Layer {layer}")
    plt.xlabel("LDA axis 0")
    plt.ylabel("LDA axis 1")
    ax.set_zlabel("LDA axis 2")
    plt.savefig(f"{COVOST_PATH}/plots/cloudmap_{subset}_lda_{layer}.png", dpi=400, facecolor='white', bbox_inches='tight')
    plt.show()

def plot_2d(layer, subset):
    axis_origin = np.zeros(1)
    axis_directions = []
    axis_directions.append(get_direction(subset, 0, layer))
    axis_directions.append(get_direction(subset, 1, layer))
    axis_directions.append(get_direction(subset, 2, layer))
    axis_directions = np.stack(axis_directions, axis=-1)
    axis_directions[:, 0] = axis_directions[:, 0] / np.linalg.norm(axis_directions[:, 0])
    axis_directions[:, 1] = get_orthonormal(axis_directions[:, 1], axis_directions[:, [0]])
    axis_directions[:, 2] = get_orthonormal(axis_directions[:, 2], axis_directions[:, :2])
    print("Getting points.")
    plot_langs = LANGs
    reps_dict = dict() # Map languages to representations.
    for lang in plot_langs:
        print("Getting points for lang: {}".format(lang))
        reps = get_reps_with_token_info(subset, lang, layer, 128)
        reps_dict[lang] = reps

    print("Plotting on axis 0 and 1.")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    # generate 21 different colors, one for each language
    # the first 4 colors need to be in red colors, the 5th-11th colors need to be in bleu colors, the rest colors need to be in green colors
    colors = ["#FF0000", "#FF3300", "#FF8080", "#FF6666", "#99CCFF", "#0000FF", "#3333FF", "#8080FF", "#6699FF", "#0000CC",  "#FFFF00", "#FFFF80", "#FFCC00", "#FFFF33", "#FFCC66", "#FFCC33", "#FFCC99", "#FFFF66", "#FFCC00", "#FF9933", "#FFCC33"]

    # axis 0 and axis 1
    for i, (label, reps) in enumerate(reps_dict.items()):
        projected_points = project_reps(axis_directions, reps, axis_origin)
        marker = 'o'
        if len(LANGs) == 21:
            if LANGs[i] in HIGHs:
                marker = '.'
            if LANGs[i] in LOWs:
                marker = '2'
            if LANGs[i] in MIDs:
                marker = '+'
        ax.scatter(projected_points[:, 0], projected_points[:, 1], label=label, c=colors[i], marker=marker, alpha=1.0, s=20)
    # legend = plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    # for lh in legend.legendHandles:
    #     lh.set_alpha(1.0)
    #     lh.set_sizes([50.0])

    plt.title(f"Layer {layer}")
    plt.xlabel("LDA axis 0")
    plt.ylabel("LDA axis 1")
    plt.savefig(f"{COVOST_PATH}/plots/cloudmap_{subset}_lda_{layer}_axis0-1.png", dpi=600, facecolor='white', bbox_inches='tight')
    # plt.show()

    print("Plotting on axis 1 and 2.")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    # generate 21 different colors, one for each language
    # the first 4 colors need to be in red colors, the 5th-11th colors need to be in bleu colors, the rest colors need to be in green colors
    colors = ["#FF0000", "#FF3300", "#FF8080", "#FF6666", "#99CCFF", "#0000FF", "#3333FF", "#8080FF", "#6699FF", "#0000CC",  "#FFFF00", "#FFFF80", "#FFCC00", "#FFFF33", "#FFCC66", "#FFCC33", "#FFCC99", "#FFFF66", "#FFCC00", "#FF9933", "#FFCC33"]

    # axis 1 and axis 2
    for i, (label, reps) in enumerate(reps_dict.items()):
        projected_points = project_reps(axis_directions, reps, axis_origin)
        marker = 'o'
        if len(LANGs) == 21:
            if LANGs[i] in HIGHs:
                marker = '.'
            if LANGs[i] in LOWs:
                marker = '2'
            if LANGs[i] in MIDs:
                marker = '+'
        ax.scatter(projected_points[:, 1], projected_points[:, 2], label=label, c=colors[i], marker=marker, alpha=1.0, s=20)
    # legend = plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    # for lh in legend.legendHandles:
    #     lh.set_alpha(1.0)
    #     lh.set_sizes([50.0])

    plt.title(f"Layer {layer}")
    plt.xlabel("LDA axis 1")
    plt.ylabel("LDA axis 2")
    plt.savefig(f"{COVOST_PATH}/plots/cloudmap_{subset}_lda_{layer}_axis1-2.png", dpi=600, facecolor='white', bbox_inches='tight')
    # plt.show()


    print("Plotting on axis 0 and 2.")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    # generate 21 different colors, one for each language
    # the first 4 colors need to be in red colors, the 5th-11th colors need to be in bleu colors, the rest colors need to be in green colors
    colors = ["#FF0000", "#FF3300", "#FF8080", "#FF6666", "#99CCFF", "#0000FF", "#3333FF", "#8080FF", "#6699FF", "#0000CC",  "#FFFF00", "#FFFF80", "#FFCC00", "#FFFF33", "#FFCC66", "#FFCC33", "#FFCC99", "#FFFF66", "#FFCC00", "#FF9933", "#FFCC33"]

    # axis 0 and axis 2
    for i, (label, reps) in enumerate(reps_dict.items()):
        projected_points = project_reps(axis_directions, reps, axis_origin)
        marker = 'o'
        if len(LANGs) == 21:
            if LANGs[i] in HIGHs:
                marker = '.'
            if LANGs[i] in LOWs:
                marker = '2'
            if LANGs[i] in MIDs:
                marker = '+'
        ax.scatter(projected_points[:, 0], projected_points[:, 2], label=label, c=colors[i], marker=marker, alpha=1.0, s=20)
    # legend = plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    # for lh in legend.legendHandles:
    #     lh.set_alpha(1.0)
    #     lh.set_sizes([50.0])

    plt.title(f"Layer {layer}")
    plt.xlabel("LDA axis 0")
    plt.ylabel("LDA axis 2")
    plt.savefig(f"{COVOST_PATH}/plots/cloudmap_{subset}_lda_{layer}_axis0-2.png", dpi=600, facecolor='white', bbox_inches='tight')

if __name__ == "__main__":
    subset = "test"
    # layer=11
    for layer in [11]:
        np.random.seed(42)
        plot_3d(layer, subset)
        plot_2d(layer, subset)