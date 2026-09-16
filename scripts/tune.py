#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader

from merge_emotion.config import compose_config, save_yaml
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.trainer import train
from merge_emotion.models.multimodal import EmotionAlignmentModel
from merge_emotion.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--config", default="configs/pipeline_experiments/multimodal_noalign.yaml")
    args = p.parse_args()
    base = compose_config(ROOT / args.config, ROOT)
    study_path = ROOT / "studies/optuna_shared.db"; study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///%s" % study_path

    def objective(trial):
        cfg = compose_config(ROOT / args.config, ROOT)
        cfg["optimizer"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        cfg["optimizer"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.40)
        cfg["audio_model"]["dropout"] = dropout; cfg["lyrics_model"]["dropout"] = dropout; cfg["multimodal"]["dropout"] = dropout
        cfg["emotion_loss"]["lambda_uni"] = trial.suggest_float("lambda_uni", 0.1, 1.0)
        cfg["training"]["max_epochs"] = min(25, int(cfg["training"]["max_epochs"]))
        seed_everything(int(cfg["project"]["seed"]), True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modality = cfg["experiment"]["modality"]
        manifest = ROOT / cfg["dataset"]["manifest"]
        audio_cache = Path(cfg["paths"]["cache_root"]) / "audio/mert_v1_95m.pt"; lyrics_cache = Path(cfg["paths"]["cache_root"]) / "lyrics/roberta_base.pt"
        train_set = MergeFeatureDataset(manifest, "train", audio_cache, lyrics_cache, modality); val_set = MergeFeatureDataset(manifest, "validation", audio_cache, lyrics_cache, modality)
        common = dict(batch_size=int(cfg["training"]["batch_size"]), num_workers=int(cfg["runtime"]["num_workers"]), pin_memory=bool(cfg["runtime"]["pin_memory"]))
        train_loader = DataLoader(train_set, shuffle=True, **common); val_loader = DataLoader(val_set, shuffle=False, **common)
        model = EmotionAlignmentModel(cfg["audio_model"], cfg["lyrics_model"], cfg["multimodal"]).to(device)
        trial_dir = ROOT / "results/runs/tuning" / ("trial_%04d" % trial.number); ckpt = ROOT / "checkpoints/tuning" / ("trial_%04d" % trial.number)
        _, _, best = train(model, train_loader, val_loader, device, cfg, trial_dir, ckpt)
        return best

    study = optuna.create_study(direction="maximize", study_name="shared_hyperparameters", storage=storage, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=2026))
    study.optimize(objective, n_trials=args.trials)
    best = base.copy(); best.setdefault("tuning", {})["best_params"] = study.best_params; best["tuning"]["best_value"] = study.best_value
    output = ROOT / "results/manifests/optuna_best.yaml"; save_yaml(best["tuning"], output)
    print("Best validation macro-F1:", study.best_value); print("Best parameters:", study.best_params); print("Saved", output)


if __name__ == "__main__":
    main()
