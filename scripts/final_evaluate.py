#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from merge_emotion.config import load_yaml
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.checkpoint import load_checkpoint
from merge_emotion.engine.evaluator import evaluate
from merge_emotion.models.multimodal import EmotionAlignmentModel
from merge_emotion.reproducibility import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Explicit one-time held-out test evaluation.")
    p.add_argument("--selection-manifest", required=True)
    p.add_argument("--index", type=int, default=None, help="Evaluate one selected entry; omit to evaluate all sequentially")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    manifest_path = Path(args.selection_manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    selection = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not selection.get("frozen", False):
        raise ValueError("Selection manifest is not frozen; refusing test evaluation")
    entries = selection.get("runs", [])
    if args.index is not None:
        entries = [entries[args.index]]
    if not entries:
        raise ValueError("Selection manifest contains no runs")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for entry in entries:
        run_id = str(entry["run_id"])
        run_dir = Path(entry["run_dir"])
        checkpoint = Path(entry["checkpoint"])
        if sha256_file(checkpoint) != str(entry["checkpoint_sha256"]):
            raise ValueError("Checkpoint hash mismatch for %s" % run_id)
        cfg = load_yaml(run_dir / "config.yaml")
        modality = cfg["experiment"]["modality"]
        dataset = MergeFeatureDataset(
            PROJECT_ROOT / cfg["dataset"]["manifest"],
            "test",
            Path(cfg["paths"]["cache_root"]) / "audio" / "mert_v1_95m.pt",
            Path(cfg["paths"]["cache_root"]) / "lyrics" / "roberta_base.pt",
            modality,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["runtime"]["num_workers"]),
            pin_memory=bool(cfg["runtime"]["pin_memory"] and device.type == "cuda"),
        )
        model = EmotionAlignmentModel(cfg["audio_model"], cfg["lyrics_model"], cfg["multimodal"]).to(device)
        load_checkpoint(checkpoint, model, restore_rng=False)
        metrics, predictions = evaluate(model, loader, device, modality)
        metrics.update(
            {
                "run_id": run_id,
                "experiment": cfg["experiment"]["name"],
                "seed": int(cfg["project"]["seed"]),
                "split": "test",
                "selection_manifest": str(manifest_path),
                "checkpoint_sha256": entry["checkpoint_sha256"],
            }
        )
        out_dir = PROJECT_ROOT / "results" / "final_test" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(metrics, out_dir / "metrics.json")
        pd.DataFrame(predictions).to_csv(out_dir / "predictions.csv", index=False)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
