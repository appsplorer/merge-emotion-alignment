from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureStore:
    def __init__(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.ids = list(payload["ids"])
        self.features = payload["features"].float()
        self.mask = payload["mask"].bool()
        self.index = {sid: i for i, sid in enumerate(self.ids)}
        if self.features.shape[:2] != self.mask.shape:
            raise ValueError("Feature and mask shapes are inconsistent in %s" % path)

    def get(self, item_id: str):
        idx = self.index[item_id]
        return self.features[idx], self.mask[idx]


class MergeFeatureDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, audio_cache: Path, lyrics_cache: Path, modality: str = "multimodal"):
        frame = pd.read_csv(manifest_path)
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        self.modality = modality
        self.audio = FeatureStore(audio_cache) if modality in {"audio", "multimodal"} else None
        self.lyrics = FeatureStore(lyrics_cache) if modality in {"lyrics", "multimodal"} else None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        item: Dict[str, object] = {
            "song_id": str(row["song_id"]),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "quadrant": str(row["quadrant"]),
        }
        if self.audio is not None:
            feat, mask = self.audio.get(str(row["audio_song"]))
            item["audio_features"] = feat
            item["audio_mask"] = mask
        if self.lyrics is not None:
            feat, mask = self.lyrics.get(str(row["lyric_song"]))
            item["lyrics_features"] = feat
            item["lyrics_mask"] = mask
        return item
