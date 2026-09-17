#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from merge_emotion.config import load_yaml
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.checkpoint import load_checkpoint
from merge_emotion.engine.evaluator import move_batch
from merge_emotion.models.multimodal import EmotionAlignmentModel
from merge_emotion.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parents[1]

SELECTION = (
    ROOT
    / "results"
    / "manifests"
    / "final_selection.yaml"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "representations"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_frozen_run(
    experiment: str,
    seed: int,
):
    selection = load_yaml(
        SELECTION
    )

    if not selection.get(
        "frozen",
        False,
    ):
        raise RuntimeError(
            "Final selection manifest is not frozen"
        )

    matches = [
        row
        for row
        in selection["runs"]
        if (
            row["experiment"]
            == experiment
            and int(
                row["seed"]
            )
            == int(seed)
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one frozen %s seed=%d run; found %d"
            % (
                experiment,
                seed,
                len(matches),
            )
        )

    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        choices=(
            "multimodal_noalign",
            "multimodal_fixed",
            "cspa_affect",
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.require_cuda
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is required"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    frozen = find_frozen_run(
        args.experiment,
        args.seed,
    )

    run_dir = Path(
        frozen["run_dir"]
    )

    checkpoint = Path(
        frozen["checkpoint"]
    )

    expected_sha = str(
        frozen["checkpoint_sha256"]
    )

    observed_sha = sha256_file(
        checkpoint
    )

    if observed_sha != expected_sha:
        raise RuntimeError(
            "Frozen checkpoint hash mismatch"
        )

    config = load_yaml(
        run_dir
        / "config.yaml"
    )

    seed_everything(
        int(args.seed),
        bool(
            config[
                "project"
            ].get(
                "deterministic",
                True,
            )
        ),
    )

    dataset = MergeFeatureDataset(
        ROOT
        / config[
            "dataset"
        ][
            "manifest"
        ],
        "validation",
        Path(
            config[
                "paths"
            ][
                "cache_root"
            ]
        )
        / "audio"
        / "mert_v1_95m.pt",
        Path(
            config[
                "paths"
            ][
                "cache_root"
            ]
        )
        / "lyrics"
        / "roberta_base.pt",
        "multimodal",
    )

    loader = DataLoader(
        dataset,
        batch_size=int(
            config[
                "training"
            ][
                "batch_size"
            ]
        ),
        shuffle=False,
        num_workers=int(
            config[
                "runtime"
            ][
                "num_workers"
            ]
        ),
        pin_memory=bool(
            config[
                "runtime"
            ][
                "pin_memory"
            ]
            and device.type
            == "cuda"
        ),
    )

    model = EmotionAlignmentModel(
        config["audio_model"],
        config["lyrics_model"],
        config["multimodal"],
    ).to(device)

    load_checkpoint(
        checkpoint,
        model,
        restore_rng=False,
    )

    model.eval()

    song_ids = []
    labels = []

    audio_rows = []
    lyrics_rows = []
    multimodal_rows = []

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(
                batch,
                device,
            )

            outputs = model(
                batch,
                modality="multimodal",
            )

            song_ids.extend(
                [
                    str(value)
                    for value
                    in batch[
                        "song_id"
                    ]
                ]
            )

            labels.extend(
                batch[
                    "label"
                ]
                .detach()
                .cpu()
                .tolist()
            )

            audio_rows.append(
                outputs[
                    "audio_repr"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            lyrics_rows.append(
                outputs[
                    "lyrics_repr"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            multimodal_rows.append(
                outputs[
                    "multimodal_repr"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

    audio = np.concatenate(
        audio_rows,
        axis=0,
    )

    lyrics = np.concatenate(
        lyrics_rows,
        axis=0,
    )

    multimodal = np.concatenate(
        multimodal_rows,
        axis=0,
    )

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    ids_array = np.asarray(
        song_ids,
        dtype=str,
    )

    n = len(ids_array)

    if not (
        n
        == 332
        == len(labels_array)
        == audio.shape[0]
        == lyrics.shape[0]
        == multimodal.shape[0]
    ):
        raise RuntimeError(
            "Expected exactly 332 validation examples; "
            "received ids=%d audio=%d lyrics=%d multimodal=%d"
            % (
                n,
                audio.shape[0],
                lyrics.shape[0],
                multimodal.shape[0],
            )
        )

    for name, matrix in (
        ("audio", audio),
        ("lyrics", lyrics),
        ("multimodal", multimodal),
    ):
        if not np.isfinite(
            matrix
        ).all():
            raise RuntimeError(
                "%s embeddings contain non-finite values"
                % name
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        "%s_validation_seed%d"
        % (
            args.experiment,
            args.seed,
        )
    )

    output = (
        OUTPUT_ROOT
        / (
            stem
            + ".npz"
        )
    )

    np.savez_compressed(
        output,
        song_id=ids_array,
        label=labels_array,
        audio_repr=audio,
        lyrics_repr=lyrics,
        multimodal_repr=multimodal,
    )

    metadata = {
        "experiment":
            args.experiment,

        "seed":
            int(args.seed),

        "split":
            "validation",

        "n":
            int(n),

        "checkpoint":
            str(checkpoint),

        "checkpoint_sha256":
            observed_sha,

        "source_run_id":
            str(
                frozen["run_id"]
            ),

        "test_used":
            False,

        "output":
            str(output),
    }

    metadata_path = (
        OUTPUT_ROOT
        / (
            stem
            + ".json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        "audio",
        audio.shape,
        "lyrics",
        lyrics.shape,
        "multimodal",
        multimodal.shape,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
