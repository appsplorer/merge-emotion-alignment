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

from merge_emotion.config import (
    apply_overrides,
    apply_tuned_params,
    compose_config,
    load_yaml,
    save_yaml,
)
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.checkpoint import load_checkpoint
from merge_emotion.engine.evaluator import evaluate
from merge_emotion.engine.trainer import train
from merge_emotion.models.multimodal import EmotionAlignmentModel
from merge_emotion.reproducibility import seed_everything, system_metadata, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_tuned_params(path: Path):
    data = load_yaml(path)
    return data.get("best_params", data)


def _resolve_resume(value: str, checkpoint_dir: Path):
    if not value:
        return None
    if value == "auto":
        candidate = checkpoint_dir / "last.pt"
        return candidate if candidate.exists() else None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    p = argparse.ArgumentParser(description="Train using train/validation only; test is hard-firewalled.")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--tuned", default=None, help="YAML produced by scripts/tune.py")
    p.add_argument("--run-id", default=None, help="Stable run id; useful for resumable Slurm jobs")
    p.add_argument("--resume", default=None, help="Checkpoint path or 'auto' for <run>/last.pt")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    config = compose_config(PROJECT_ROOT / args.config, PROJECT_ROOT)
    if args.tuned:
        tuned_path = Path(args.tuned)
        if not tuned_path.is_absolute():
            tuned_path = PROJECT_ROOT / tuned_path
        config = apply_tuned_params(config, _load_tuned_params(tuned_path))
    config = apply_overrides(config, args.set)

    seed = int(config["project"]["seed"])
    seed_everything(seed, bool(config["project"].get("deterministic", True)))
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this production run")
    if config["runtime"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["runtime"]["device"])

    name = config["experiment"]["name"]
    run_id = args.run_id or "%s_seed%d_%s" % (
        name,
        seed,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    run_dir = Path(config["paths"]["run_root"]) / run_id
    checkpoint_dir = Path(config["paths"]["checkpoint_root"]) / run_id
    resume_path = _resolve_resume(args.resume, checkpoint_dir)

    if run_dir.exists() and resume_path is None:
        raise FileExistsError(
            "%s already exists. Use --resume auto (or choose a different --run-id)." % run_dir
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    if resume_path is not None and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                print(metrics_path.read_text(encoding="utf-8"))
            else:
                print(json.dumps(status, indent=2))
            return

    if resume_path is None:
        save_yaml(config, run_dir / "config.yaml")
        write_json(system_metadata(), run_dir / "system.json")
    else:
        saved_config = load_yaml(run_dir / "config.yaml")
        if saved_config != config:
            raise ValueError("Resolved config differs from the existing run config; refusing resume")

    modality = config["experiment"]["modality"]
    manifest = PROJECT_ROOT / config["dataset"]["manifest"]
    audio_cache = Path(config["paths"]["cache_root"]) / "audio" / "mert_v1_95m.pt"
    lyrics_cache = Path(config["paths"]["cache_root"]) / "lyrics" / "roberta_base.pt"
    train_set = MergeFeatureDataset(manifest, "train", audio_cache, lyrics_cache, modality)
    val_set = MergeFeatureDataset(manifest, "validation", audio_cache, lyrics_cache, modality)
    common = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=bool(config["runtime"]["pin_memory"] and device.type == "cuda"),
    )
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    model = EmotionAlignmentModel(
        config["audio_model"], config["lyrics_model"], config["multimodal"]
    ).to(device)

    started = time.time()
    history_new, gradients_new, best = train(
        model,
        train_loader,
        val_loader,
        device,
        config,
        run_dir,
        checkpoint_dir,
        resume_checkpoint=resume_path,
    )

    def append_csv(path: Path, rows):
        if not rows:
            return
        current = pd.read_csv(path) if path.exists() else pd.DataFrame()
        merged = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        if "epoch" in merged.columns:
            merged = merged.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
        elif "step" in merged.columns:
            merged = merged.drop_duplicates(subset=["step"], keep="last").sort_values("step")
        merged.to_csv(path, index=False)

    append_csv(run_dir / "history.csv", history_new)
    append_csv(run_dir / "gradient_stats.csv", gradients_new)

    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Training produced no best checkpoint")
    load_checkpoint(best_path, model, restore_rng=False)
    val_metrics, val_predictions = evaluate(model, val_loader, device, modality)
    val_metrics.update(
        {
            "run_id": run_id,
            "experiment": name,
            "stage": config.get("experiment", {}).get("stage", "development"),
            "seed": seed,
            "split": "validation",
            "best_val_macro_f1": float(best),
            "wall_seconds": time.time() - started,
            "test_evaluated": False,
        }
    )
    write_json(val_metrics, run_dir / "metrics.json")
    pd.DataFrame(val_predictions).to_csv(run_dir / "validation_predictions.csv", index=False)
    write_json({"status": "completed", "run_id": run_id}, run_dir / "status.json")
    print(json.dumps(val_metrics, indent=2))


if __name__ == "__main__":
    main()
