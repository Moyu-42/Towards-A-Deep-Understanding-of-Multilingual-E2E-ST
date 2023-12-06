# Towards a Deep Understanding of Multilingual End-to-End Speech Translation

This is the offical source code for EMNLP2023-findings paper "Towards a Deep Understanding of Multilingual End-to-End Speech Translation".

### Requirements and Installation

- [PyTorch](http://pytorch.org/) version = 1.13.0
- Python version = 3.8.8
- [Fairseq](https://github.com/facebookresearch/fairseq) version = 0.12.2

Note: This is the pip version we used in our experiments, other version may work well.

### 0. LASER-mined evaluation datasets

Download Common Voice V13 dataset from [Common Voice](https://commonvoice.mozilla.org/en/datasets) for ar, cy, et, id, ja, lv, mn, sl, sv-SE, ta, tr

Download [LASER](https://github.com/facebookresearch/LASER/tree/main) with "laser3-cym_Latn.v1.pt" "laser3-ind_Latn.v1.pt" "laser3-khk_Cyrl.v1.pt" "laser3-tam_Taml.v1.pt" and "laser3-tur_Latn.v1.pt" encoders

```bash
python3 tools/get_v13_not_v4.py
python3 tools/v13_dataset.py --data-root CV13_PATH --src-lang lang # generate Common Voice 13 test file
bash scripts/CoVoST/mwp_laser_m2o.py # get laser embedding and bitext
python3 tools/mwp.py # extract evaluation datasets
```

### 1. Train Multilingual End-to-End Speech Translation Models

preprocessing the CoVoST 2 data following [facebookresearch/covost](https://github.com/facebookresearch/covost)

Train ASR models

```bash
bash scripts/CoVoST/asr_multitrain.sh
```

Train ST models

```bash
bash scripts/CoVoST/st_multitrain.sh
bash scripts/CoVoST/st_generate.sh
```

Using average_checkpoints.py for last 5 epoches 

```bash
python scripts/average_checkpoints.py --inputs INPUTs --output INPUTs/avg.pt --num-epoch-checkpoints 5
```

### 2. SVCCA scores

```bash
bash scripts/CoVoST/get_representations.sh # get encoder representations for X-En pairs
bash scripts/CoVoST/get_decoder_representations.sh # get decoder representations for En-X pairs
```

get SVCCA scores for sentence-level representations

```bash
python3 scripts/CoVoST/get_svcca_scores.py
```

### 3. Ploting

```bash
python3 scripts/CoVoST/ploting/cca_clustermap_plot.py # for Figure 1 and 3
python3 scripts/CoVoST/ploting/get_cloudmap.py # for Figure 2
```

### Citation

Please cite as:

```
@article{sun2023towards,
  title={Towards a Deep Understanding of Multilingual End-to-End Speech Translation},
  author={Sun, Haoran and Zhao, Xiaohu and Lei, Yikun and Zhu, Shaolin and Xiong, Deyi},
  journal={arXiv preprint arXiv:2310.20456},
  year={2023}
}
```

Part of the codes are adopted from:

```
@article{raghu2017svcca,
  title={Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability},
  author={Raghu, Maithra and Gilmer, Justin and Yosinski, Jason and Sohl-Dickstein, Jascha},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

```
@article{chang2022geometry,
  title={The geometry of multilingual language model representations},
  author={Chang, Tyler A and Tu, Zhuowen and Bergen, Benjamin K},
  journal={arXiv preprint arXiv:2205.10964},
  year={2022}
}
```

