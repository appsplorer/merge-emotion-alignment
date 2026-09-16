from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .integrity import (
    AUDIO_EXTENSIONS,
    AUDIO_ID_ALIASES,
    ID_ALIASES,
    LABEL_ALIASES,
    LYRIC_EXTENSIONS,
    LYRIC_ID_ALIASES,
    clean_identifier,
    discover_metadata_file,
    discover_split_files,
    infer_column,
    media_index,
    normalize_quadrant,
    read_table,
    select_metadata_id_column,
)

LABEL_TO_INDEX = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}


def _resolve_path(index: Dict[str, List[str]], candidates: List[str], kind: str) -> str:
    tried = []
    for key in candidates:
        if not key or key in tried:
            continue
        tried.append(key)
        paths = index.get(key, [])
        if len(paths) == 1:
            return paths[0]
        if len(paths) > 1:
            raise ValueError("Multiple %s files match %s" % (kind, key))
    raise ValueError("No unique %s file found for candidate IDs %s" % (kind, tried))


def build_pair_manifest(dataset_root: Path, profile: str, output_path: Path) -> pd.DataFrame:
    _, metadata = discover_metadata_file(dataset_root)
    split_files = discover_split_files(dataset_root, profile)
    all_split_ids = set()
    split_by_id = {}
    split_label_by_id = {}
    for split, path in split_files.items():
        frame = read_table(path)
        id_col = infer_column(frame, ID_ALIASES)
        label_col = infer_column(frame, LABEL_ALIASES)
        ids = frame[id_col].map(clean_identifier)
        labels = normalize_quadrant(frame[label_col])
        for sid, label in zip(ids, labels):
            all_split_ids.add(sid)
            split_by_id[sid] = split
            split_label_by_id[sid] = label
    split_id_col = select_metadata_id_column(metadata, all_split_ids)
    audio_col = infer_column(metadata, AUDIO_ID_ALIASES)
    lyric_col = infer_column(metadata, LYRIC_ID_ALIASES)
    label_col = infer_column(metadata, LABEL_ALIASES)
    audio_index = media_index(dataset_root, AUDIO_EXTENSIONS)
    lyric_index = media_index(dataset_root, LYRIC_EXTENSIONS)
    rows = []
    for _, row in metadata.iterrows():
        split_id = clean_identifier(row[split_id_col])
        if split_id not in split_by_id:
            continue
        audio_id = clean_identifier(row[audio_col])
        lyric_id = clean_identifier(row[lyric_col])
        label = normalize_quadrant(pd.Series([row[label_col]])).iloc[0]
        if split_label_by_id[split_id] != label:
            raise ValueError("Label mismatch for %s" % split_id)
        rows.append({
            "song_id": split_id,
            "audio_song": audio_id,
            "lyric_song": lyric_id,
            "audio_path": _resolve_path(audio_index, [audio_id, split_id, lyric_id], "audio"),
            "lyric_path": _resolve_path(lyric_index, [lyric_id, split_id, audio_id], "lyrics"),
            "quadrant": label,
            "label": LABEL_TO_INDEX[label],
            "split": split_by_id[split_id],
        })
    manifest = pd.DataFrame(rows).sort_values(["split", "song_id"]).reset_index(drop=True)
    if len(manifest) != len(metadata):
        raise ValueError("Manifest has %d rows; expected %d" % (len(manifest), len(metadata)))
    if manifest["song_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate song IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest
