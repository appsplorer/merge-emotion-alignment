#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import torch

from merge_emotion.analysis.small_extension import (
    classify_collision,
    clean_identifier,
    cosine_similarity,
    mean_cached_representation,
    normalize_identity_part,
    normalize_lyrics_text,
    sha256_file,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]

INTEGRITY = (
    ROOT
    / "results"
    / "manifests"
    / "merge_70-15-15_integrity.json"
)

PAIR_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "pairs_70-15-15.csv"
)

AUDIO_CACHE = (
    ROOT
    / "cache"
    / "audio"
    / "mert_v1_95m.pt"
)

LYRICS_CACHE = (
    ROOT
    / "cache"
    / "lyrics"
    / "roberta_base.pt"
)

OUTPUT = (
    ROOT
    / "results"
    / "tables"
    / "artist_title_collision_audit.csv"
)


def resolve_path(value) -> Path:
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def normalized_row_values(row):
    values = set()

    for value in row.tolist():
        text = normalize_identity_part(value)

        if text:
            values.add(text)

    return values


def raw_row_identifiers(row):
    values = set()

    for value in row.tolist():
        text = clean_identifier(value)

        if text:
            values.add(text)

    return values


def main() -> int:
    report = json.loads(
        INTEGRITY.read_text(encoding="utf-8")
    )

    metadata_path = Path(report["metadata_file"])

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = pd.read_csv(metadata_path)
    pairs = pd.read_csv(PAIR_MANIFEST)

    print("Metadata columns:")
    for column in metadata.columns:
        print(" -", column)

    pair_lookup = {}

    for index, row in pairs.iterrows():
        for column in (
            "song_id",
            "audio_song",
            "lyric_song",
        ):
            identifier = clean_identifier(row[column])

            pair_lookup.setdefault(
                identifier,
                set(),
            ).add(index)

    audio_payload = torch.load(
        AUDIO_CACHE,
        map_location="cpu",
        weights_only=False,
    )

    lyric_payload = torch.load(
        LYRICS_CACHE,
        map_location="cpu",
        weights_only=False,
    )

    output_rows = []

    candidates = report.get(
        "artist_title_cross_split_collisions",
        [],
    )

    print()
    print(
        "Collision candidates:",
        len(candidates),
    )

    for candidate in candidates:
        identity = candidate["identity"]

        if "::" not in identity:
            raise RuntimeError(
                "Unexpected collision identity: %s"
                % identity
            )

        artist_raw, title_raw = identity.split(
            "::",
            1,
        )

        artist = normalize_identity_part(
            artist_raw
        )

        title = normalize_identity_part(
            title_raw
        )

        resolved_pair_indexes = set()

        for _, metadata_row in metadata.iterrows():
            normalized_values = normalized_row_values(
                metadata_row
            )

            if (
                artist not in normalized_values
                or title not in normalized_values
            ):
                continue

            identifiers = raw_row_identifiers(
                metadata_row
            )

            for identifier in identifiers:
                resolved_pair_indexes.update(
                    pair_lookup.get(
                        identifier,
                        set(),
                    )
                )

        resolved = (
            pairs.loc[
                sorted(resolved_pair_indexes)
            ]
            .drop_duplicates("song_id")
            .copy()
        )

        if len(resolved) < 2:
            raise RuntimeError(
                "Could not resolve at least two paired rows for %s. "
                "Resolved rows=%d"
                % (
                    identity,
                    len(resolved),
                )
            )

        cross_split_pairs = []

        records = resolved.to_dict(
            orient="records"
        )

        for left, right in itertools.combinations(
            records,
            2,
        ):
            if str(left["split"]) != str(right["split"]):
                cross_split_pairs.append(
                    (
                        left,
                        right,
                    )
                )

        if not cross_split_pairs:
            raise RuntimeError(
                "No cross-split row pair recovered for %s"
                % identity
            )

        for left, right in cross_split_pairs:
            audio_a = resolve_path(
                left["audio_path"]
            )

            audio_b = resolve_path(
                right["audio_path"]
            )

            lyric_a = resolve_path(
                left["lyric_path"]
            )

            lyric_b = resolve_path(
                right["lyric_path"]
            )

            audio_sha_a = sha256_file(
                audio_a
            )

            audio_sha_b = sha256_file(
                audio_b
            )

            lyric_text_a = normalize_lyrics_text(
                lyric_a.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            lyric_text_b = normalize_lyrics_text(
                lyric_b.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            lyrics_sha_a = sha256_text(
                lyric_text_a
            )

            lyrics_sha_b = sha256_text(
                lyric_text_b
            )

            audio_repr_a = mean_cached_representation(
                audio_payload,
                clean_identifier(
                    left["audio_song"]
                ),
            )

            audio_repr_b = mean_cached_representation(
                audio_payload,
                clean_identifier(
                    right["audio_song"]
                ),
            )

            lyrics_repr_a = mean_cached_representation(
                lyric_payload,
                clean_identifier(
                    left["lyric_song"]
                ),
            )

            lyrics_repr_b = mean_cached_representation(
                lyric_payload,
                clean_identifier(
                    right["lyric_song"]
                ),
            )

            audio_cosine = cosine_similarity(
                audio_repr_a,
                audio_repr_b,
            )

            lyrics_cosine = cosine_similarity(
                lyrics_repr_a,
                lyrics_repr_b,
            )

            audio_equal = (
                audio_sha_a
                == audio_sha_b
            )

            lyrics_equal = (
                lyrics_sha_a
                == lyrics_sha_b
            )

            verdict = classify_collision(
                audio_sha_equal=audio_equal,
                lyrics_normalized_equal=lyrics_equal,
                audio_cosine=audio_cosine,
                lyrics_cosine=lyrics_cosine,
            )

            output_rows.append(
                {
                    "identity":
                        identity,

                    "song_id_a":
                        clean_identifier(
                            left["song_id"]
                        ),

                    "split_a":
                        str(
                            left["split"]
                        ),

                    "label_a":
                        int(
                            left["label"]
                        ),

                    "quadrant_a":
                        str(
                            left["quadrant"]
                        ),

                    "song_id_b":
                        clean_identifier(
                            right["song_id"]
                        ),

                    "split_b":
                        str(
                            right["split"]
                        ),

                    "label_b":
                        int(
                            right["label"]
                        ),

                    "quadrant_b":
                        str(
                            right["quadrant"]
                        ),

                    "same_label":
                        bool(
                            int(left["label"])
                            == int(right["label"])
                        ),

                    "audio_sha256_a":
                        audio_sha_a,

                    "audio_sha256_b":
                        audio_sha_b,

                    "audio_sha_equal":
                        bool(
                            audio_equal
                        ),

                    "lyrics_normalized_sha256_a":
                        lyrics_sha_a,

                    "lyrics_normalized_sha256_b":
                        lyrics_sha_b,

                    "lyrics_normalized_equal":
                        bool(
                            lyrics_equal
                        ),

                    "cached_mert_cosine":
                        float(
                            audio_cosine
                        ),

                    "cached_roberta_cosine":
                        float(
                            lyrics_cosine
                        ),

                    "verdict":
                        verdict,
                }
            )

    output = pd.DataFrame(
        output_rows
    )

    if output.empty:
        raise RuntimeError(
            "Collision audit produced no rows"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("ARTIST/TITLE COLLISION AUDIT")
    print("=" * 78)

    print(
        output.to_string(
            index=False
        )
    )

    print()
    print(
        "Verdict counts:"
    )

    print(
        output["verdict"]
        .value_counts()
        .to_string()
    )

    print()
    print("Saved:", OUTPUT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
