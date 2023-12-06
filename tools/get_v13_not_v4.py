#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

LANGs = ["ar", "cy", "et", "id", "ja", "lv", "mn", "nl", "sl", "sv-SE", "ta", "tr"]

DATA_RAW=""

if __name__ == "__main__":
    for lang in LANGs:
        print("lang: ", lang)
        v13_data = pd.read_csv(f"{DATA_RAW}/CV_13/{lang}/validated.tsv", sep='\t')
        v4_data = pd.read_csv(f"{DATA_RAW}/CoVoST/{lang}/train.tsv", sep='\t')

        v13_not_in_v4 = v13_data[~v13_data.path.isin(v4_data.path)]
        v13_not_in_v4.to_csv(f"{DATA_RAW}/CV_13/{lang}/validated_not_in_v4.tsv", sep='\t', index=False)
        print("before: ", len(v13_data), " after: ", len(v13_not_in_v4))
        print("v4 len: ", len(v4_data))
