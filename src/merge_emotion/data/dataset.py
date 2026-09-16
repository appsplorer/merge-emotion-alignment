from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


class MergeFeatureDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, audio_cache: Path, lyrics_cache: Path, modality: str = "multimodal"):
        frame = pd.read_csv(manifest_path)
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        self.modality = modality
        self.audio = torch.load(audio_cache, map_location="cpu") if modality in {"audio", "multimodal"} else None
        self.lyrics = torch.load(lyrics_cache, map_location="cpu") if modality in {"lyrics", "multimodal"} else None
        self.audio_index = {str(k): i for i, k in enumerate(self.audio["ids"])} if self.audio else {}
        self.lyrics_index = {str(k): i for i, k in enumerate(self.lyrics["ids"])} if self.lyrics else {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        item = {"song_id": str(row["song_id"]), "label": torch.tensor(int(row["label"]), dtype=torch.long)}
        if self.audio is not None:
            i = self.audio_index[str(row["audio_song"])]
            item["audio_features"] = self.audio["features"][i]
            item["audio_mask"] = self.audio["mask"][i]
        if self.lyrics is not None:
            i = self.lyrics_index[str(row["lyric_song"])]
            item["lyrics_features"] = self.lyrics["features"][i]
            item["lyrics_mask"] = self.lyrics["mask"][i]
        return item
