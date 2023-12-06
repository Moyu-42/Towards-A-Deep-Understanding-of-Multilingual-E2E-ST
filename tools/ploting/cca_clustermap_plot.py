import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt

for layer in range(12):
    with open(f'cca_score_dec_layer_{layer}.json') as f:
        data = json.load(f)
    
    # get the max value of data
    max_value = 0
    for k1 in data.keys():
        for k2 in data[k1].keys():
            if data[k1][k2] > max_value:
                max_value = data[k1][k2]
    max_value = 1.0

    # lang_list = ["ar", "ca", "cy", "de", "es", "et", "fa", "fr", "id", "it", "ja", "lv", "mn", "nl", "pt", "ru", "sl", "sv", "ta", "tr", "zh"]
    lang_list = ["ar", "ca", "cy", "de", "et", "fa", "id", "ja", "lv", "mn", "sl", "sv", "ta", "tr", "zh"]
    lang_branch_map = {
        "ar": "Semitic",
        "ca": "Romance",
        "cy": "Celtic",
        "de": "Germanic",
        "es": "Romance",
        "et": "Finno-Ugric",
        "fa": "Indo-Iranian",
        "fr": "Romance",
        "id": "Austronesian",
        "it": "Romance",
        "ja": "Japonic",
        "lv": "Baltic",
        "mn": "Mongolic",
        "nl": "Germanic",
        "pt": "Romance",
        "ru": "Slavic",
        "sl": "Slavic",
        "sv": "Germanic",
        "ta": "Dravidian",
        "tr": "Turkic",
        "zh": "Sino-Tibetan"
    }
    lang_family_map = {
        "ar": "Afro-Asiatic",
        "ca": "Indo-European",
        "cy": "Indo-European",
        "de": "Indo-European",
        "es": "Indo-European",
        "et": "Uralic",
        "fa": "Indo-European",
        "fr": "Indo-European",
        "id": "Austronesian",
        "it": "Indo-European",
        "ja": "Japonic",
        "lv": "Indo-European",
        "mn": "Mongolic",
        "nl": "Indo-European",
        "pt": "Indo-European",
        "ru": "Indo-European",
        "sl": "Indo-European",
        "sv": "Indo-European",
        "ta": "Dravidian",
        "tr": "Turkic",
        "zh": "Sino-Tibetan"
    }

    # structure of data is a list of dictionaries, where "lang1": {"lang2": score_lang2, "lang3": score_lang3, ...}
    # change it to heatmap format, where "lang1" is the row, "lang2" is the column, and score_lang2 is the value
    heatmap_data = []
    for lang1 in lang_list:
        for lang2 in lang_list:
            if lang1 == lang2:
                heatmap_data.append({"lang1": lang1, "lang2": lang2, "score": max_value})
                continue
            src=lang1
            tgt=lang2
            if lang1 == "zh":
                src = "zh-CN"
            if lang1 == "sv":
                src = "sv-SE"
            if lang2 == "zh":
                tgt = "zh-CN"
            if lang2 == "sv":
                tgt = "sv-SE"
            heatmap_data.append({"lang1": lang1, "lang2": lang2, "score": data[src][tgt]})
    heatmap_data = pd.DataFrame(heatmap_data)
    # rename the zh-CN in lang1 to zh
    heatmap_data["lang1"] = heatmap_data["lang1"].replace("zh-CN", "zh")
    heatmap_data["lang2"] = heatmap_data["lang2"].replace("zh-CN", "zh")
    heatmap_data["lang1"] = heatmap_data["lang1"].replace("sv-SE", "sv")
    heatmap_data["lang2"] = heatmap_data["lang2"].replace("sv-SE", "sv")
    
    # sort by branch
    heatmap_data["branch1"] = heatmap_data["lang1"].map(lang_branch_map)
    heatmap_data["branch2"] = heatmap_data["lang2"].map(lang_branch_map)
    heatmap_data = heatmap_data.sort_values(["branch1", "branch2"])
    heatmap_data['family1'] = heatmap_data['lang1'].map(lang_family_map)
    heatmap_data['family2'] = heatmap_data['lang2'].map(lang_family_map)
    # min-max normalization
    # heatmap_data["score"] = (heatmap_data["score"] - heatmap_data["score"].min()) / (heatmap_data["score"].max() - heatmap_data["score"].min())
    # desired_order = ["pt", "es", "it", "ca", "fr", "de", "nl", "sv", "cy", "ru", "sl", "lv", "fa", "zh", "tr", "mn", "et", "ta", "ar", "ja", "id", ]
    desired_order = ["ca", "de", "sv", "cy", "sl", "lv", "fa", "zh", "tr", "mn", "et", "ta", "ar", "ja", "id", ]
    df_reordered = heatmap_data.pivot("lang1", "lang2", "score").reindex(desired_order, axis=0).reindex(desired_order, axis=1)

    # a larger figure
    plt.figure(figsize=(10, 16), dpi=400)
    # larger font size
    sns.set(font_scale=1.5)

    # list of branch name
    branch_count = []
    for lang in desired_order:
        branch = lang_branch_map[lang]
        if branch not in branch_count:
            branch_count.append(branch)

    # Create a categorical palette to identify the networks
    lb_pal = sns.husl_palette(14, s=.85)
    lb_lut = dict(zip(map(str, branch_count), lb_pal))

    lf_pal = sns.husl_palette(9, s=.85)
    lf_lut = dict(zip(map(str, ["Afro-Asiatic", "Indo-European", "Uralic", "Austronesian", "Sino-Tibetan", "Japonic", "Mongolic", "Dravidian", "Turkic"]), lf_pal))

    # Convert the palette to vectors that will be drawn on the side of the matrix
    branch1 = df_reordered.index.map(lang_branch_map)
    branch2 = df_reordered.columns.map(lang_branch_map)
    branch_colors1 = pd.Series(branch1, index=df_reordered.index).map(lb_lut)
    branch_colors2 = pd.Series(branch2, index=df_reordered.columns).map(lb_lut)
    # rename the header of branch_colors1
    branch_colors1 = branch_colors1.rename("")
    branch_colors2 = branch_colors2.rename("Lang Branch")

    family1 = df_reordered.index.map(lang_family_map)
    family2 = df_reordered.columns.map(lang_family_map)
    family_colors1 = pd.Series(family1, index=df_reordered.index).map(lf_lut)
    family_colors2 = pd.Series(family2, index=df_reordered.columns).map(lf_lut)

    # Draw the full plot
    g = sns.clustermap(df_reordered, cmap="vlag",
                        row_cluster=False, col_cluster=False,
                        row_colors=branch_colors1, col_colors=branch_colors2,
                        linewidths=.75, figsize=(12, 13)
                        )
    # remove the ticklabel of the row_colors
    g.ax_row_dendrogram.set_yticklabels([])
    g.ax_col_dendrogram.set_xticklabels([])

    # set x-axis label
    g.ax_heatmap.set_xlabel("Language 1", fontsize=20)
    # set y-axis label
    g.ax_heatmap.set_ylabel("Language 2", fontsize=20)

    # rotate y-axis 0
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0)
    # move colorbar to the left center
    g.cax.set_position([.05, .2, .03, .45])
    # add label for row_colors
    for label in branch_count:
        g.ax_row_dendrogram.bar(0, 0, color=lb_lut[label],
                                label=label, linewidth=0)
    # move this label to the upper center of the plot
    g.ax_row_dendrogram.legend(loc="upper center", ncol=5, bbox_to_anchor=(2.89, 1.22))

    plt.savefig(f"cca_dec_layer_{layer}_clustermap.png")
