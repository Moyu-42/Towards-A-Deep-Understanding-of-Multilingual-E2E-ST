#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional, Tuple
from pathlib import Path
import shutil
from fairseq.examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv
)
import torchaudio
from tqdm import tqdm

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

class CoVoST_V13(Dataset):
    def __init__(
        self,
        root: str
    ) -> None:
        self.root: Path = Path(root)
        cv_tsv_path = self.root / "validated_not_in_v4.tsv"
        assert cv_tsv_path.is_file()

        df = load_df_from_tsv(cv_tsv_path)
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError as err:
                print(err)
    
    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)

def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    # Extract features
    feature_root = root / "fbank80"
    feature_root.mkdir(exist_ok=True)

    dataset = CoVoST_V13(root)
    for waveform, sample_rate, _, _, utt_id in tqdm(dataset):
        extract_fbank_features(waveform, sample_rate, feature_root / f"{utt_id}.npy")

    zip_path = root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    print("Generating manifest...")

    manifest = {c: [] for c in MANIFEST_COLUMNS}
    dataset = CoVoST_V13(root)
    for _, _, src_utt, speaker_id, utt_id in tqdm(dataset):
        manifest["id"].append(utt_id)
        manifest["audio"].append(audio_paths[utt_id])
        manifest["n_frames"].append(audio_lengths[utt_id])
        manifest["tgt_text"].append(src_utt)
        manifest["speaker"].append(speaker_id)
    
    df = pd.DataFrame.from_dict(manifest)
    df = filter_manifest_df(df, is_train_split=False)
    save_df_to_tsv(df, root / f"test_st_{args.src_lang}_en.tsv")
    # Clean up
    shutil.rmtree(feature_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    args = parser.parse_args()

    process(args)

if __name__ == "__main__":
    main()
