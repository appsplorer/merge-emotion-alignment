#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from merge_emotion.config import apply_overrides, compose_config, save_yaml
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.checkpoint import load_checkpoint
from merge_emotion.engine.evaluator import evaluate
from merge_emotion.engine.trainer import train
from merge_emotion.models.multimodal import EmotionAlignmentModel
from merge_emotion.reproducibility import seed_everything, system_metadata, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    config = apply_overrides(compose_config(PROJECT_ROOT / args.config, PROJECT_ROOT), args.set)
    seed = int(config["project"]["seed"])
    seed_everything(seed, bool(config["project"].get("deterministic", True)))
    device = torch.device("cuda" if config["runtime"]["device"] == "auto" and torch.cuda.is_available() else ("cpu" if config["runtime"]["device"] == "auto" else config["runtime"]["device"]))
    name = config["experiment"]["name"]
    run_id = "%s_seed%d_%s" % (name, seed, datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(config["paths"]["run_root"]) / run_id
    checkpoint_dir = Path(config["paths"]["checkpoint_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_yaml(config, run_dir / "config.yaml")
    write_json(system_metadata(), run_dir / "system.json")
    modality = config["experiment"]["modality"]
    manifest = PROJECT_ROOT / config["dataset"]["manifest"]
    audio_cache = Path(config["paths"]["cache_root"]) / "audio" / "mert_v1_95m.pt"
    lyrics_cache = Path(config["paths"]["cache_root"]) / "lyrics" / "roberta_base.pt"
    train_set = MergeFeatureDataset(manifest, "train", audio_cache, lyrics_cache, modality)
    val_set = MergeFeatureDataset(manifest, "validation", audio_cache, lyrics_cache, modality)
    test_set = MergeFeatureDataset(manifest, "test", audio_cache, lyrics_cache, modality)
    common = dict(batch_size=int(config["training"]["batch_size"]), num_workers=int(config["runtime"]["num_workers"]), pin_memory=bool(config["runtime"]["pin_memory"]))
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    model = EmotionAlignmentModel(config["audio_model"], config["lyrics_model"], config["multimodal"]).to(device)
    started = time.time()
    history, gradients, best = train(model, train_loader, val_loader, device, config, run_dir, checkpoint_dir)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if gradients:
        pd.DataFrame(gradients).to_csv(run_dir / "gradient_stats.csv", index=False)
    load_checkpoint(checkpoint_dir / "best.pt", model, restore_rng=False)
    metrics, predictions = evaluate(model, test_loader, device, modality)
    metrics.update({"run_id": run_id, "experiment": name, "seed": seed, "best_val_macro_f1": best, "wall_seconds": time.time() - started})
    write_json(metrics, run_dir / "metrics.json")
    pd.DataFrame(predictions).to_csv(run_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
