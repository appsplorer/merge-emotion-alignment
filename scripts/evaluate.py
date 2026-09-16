#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
from merge_emotion.config import compose_config
from merge_emotion.data.dataset import MergeFeatureDataset
from merge_emotion.engine.checkpoint import load_checkpoint
from merge_emotion.engine.evaluator import evaluate
from merge_emotion.models.multimodal import EmotionAlignmentModel

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--split", default="test"); args=p.parse_args()
    cfg=compose_config(ROOT/args.config, ROOT); modality=cfg["experiment"]["modality"]
    ds=MergeFeatureDataset(ROOT/cfg["dataset"]["manifest"], args.split, Path(cfg["paths"]["cache_root"])/"audio/mert_v1_95m.pt", Path(cfg["paths"]["cache_root"])/"lyrics/roberta_base.pt", modality)
    dl=DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["runtime"]["num_workers"]))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=EmotionAlignmentModel(cfg["audio_model"],cfg["lyrics_model"],cfg["multimodal"]).to(device); load_checkpoint(Path(args.checkpoint),model,restore_rng=False)
    metrics,preds=evaluate(model,dl,device,modality); print(json.dumps(metrics,indent=2)); pd.DataFrame(preds).to_csv(Path(args.checkpoint).with_suffix(".%s_predictions.csv"%args.split),index=False)
if __name__=="__main__": main()
