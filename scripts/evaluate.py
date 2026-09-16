#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from merge_emotion.config import load_yaml
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.checkpoint import load_checkpoint
from merge_emotion.engine.evaluator import evaluate
from merge_emotion.models.multimodal import EmotionAlignmentModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Non-test evaluation utility. Test is available only through final_evaluate.py")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="validation", choices=["train", "validation"])
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    cfg = load_yaml(run_dir / "config.yaml")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality = cfg["experiment"]["modality"]
    dataset = MergeFeatureDataset(
        PROJECT_ROOT / cfg["dataset"]["manifest"],
        args.split,
        Path(cfg["paths"]["cache_root"]) / "audio" / "mert_v1_95m.pt",
        Path(cfg["paths"]["cache_root"]) / "lyrics" / "roberta_base.pt",
        modality,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
    )
    model = EmotionAlignmentModel(cfg["audio_model"], cfg["lyrics_model"], cfg["multimodal"]).to(device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"]["checkpoint_root"]) / run_dir.name / "best.pt"
    load_checkpoint(checkpoint, model, restore_rng=False)
    metrics, _ = evaluate(model, loader, device, modality)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
