#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import torch
import yaml
from optuna.trial import TrialState
from torch.utils.data import DataLoader

from merge_emotion.config import compose_config
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.trainer import train
from merge_emotion.models.multimodal import EmotionAlignmentModel
from merge_emotion.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--config", default="configs/pipeline_experiments/multimodal_noalign.yaml")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study_path = ROOT / "studies" / "optuna_shared.db"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///%s" % study_path

    def objective(trial):
        cfg = compose_config(ROOT / args.config, ROOT)
        cfg["optimizer"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        cfg["optimizer"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.40)
        cfg["audio_model"]["dropout"] = dropout
        cfg["lyrics_model"]["dropout"] = dropout
        cfg["multimodal"]["dropout"] = dropout
        cfg["emotion_loss"]["lambda_uni"] = trial.suggest_float("lambda_uni", 0.1, 1.0)
        cfg["training"]["max_epochs"] = min(25, int(cfg["training"]["max_epochs"]))
        cfg.setdefault("experiment", {})["stage"] = "tuning"
        seed_everything(int(cfg["project"]["seed"]), True)
        modality = cfg["experiment"]["modality"]
        manifest = ROOT / cfg["dataset"]["manifest"]
        audio_cache = Path(cfg["paths"]["cache_root"]) / "audio" / "mert_v1_95m.pt"
        lyrics_cache = Path(cfg["paths"]["cache_root"]) / "lyrics" / "roberta_base.pt"
        train_set = MergeFeatureDataset(manifest, "train", audio_cache, lyrics_cache, modality)
        val_set = MergeFeatureDataset(manifest, "validation", audio_cache, lyrics_cache, modality)
        common = dict(
            batch_size=int(cfg["training"]["batch_size"]),
            num_workers=int(cfg["runtime"]["num_workers"]),
            pin_memory=bool(cfg["runtime"]["pin_memory"] and device.type == "cuda"),
        )
        train_loader = DataLoader(train_set, shuffle=True, **common)
        val_loader = DataLoader(val_set, shuffle=False, **common)
        model = EmotionAlignmentModel(cfg["audio_model"], cfg["lyrics_model"], cfg["multimodal"]).to(device)
        trial_dir = ROOT / "results" / "runs" / "tuning" / ("trial_%04d" % trial.number)
        ckpt = ROOT / "checkpoints" / "tuning" / ("trial_%04d" % trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)
        _, _, best = train(model, train_loader, val_loader, device, cfg, trial_dir, ckpt)
        return best

    study = optuna.create_study(
        direction="maximize",
        study_name="shared_hyperparameters",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=2026),
    )
    completed_trials = sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)
    remaining_trials = max(0, int(args.trials) - completed_trials)
    if remaining_trials:
        print("Completed trials:", completed_trials, "remaining to target:", remaining_trials)
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("Target completed trials already reached:", completed_trials)
    output = ROOT / "results" / "manifests" / "optuna_best.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            {"best_value": float(study.best_value), "best_params": dict(study.best_params)},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print("Best validation macro-F1:", study.best_value)
    print("Best parameters:", study.best_params)
    print("Saved", output)


if __name__ == "__main__":
    main()
