from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, global_step, best_metric, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict() if scaler else None, "epoch": epoch, "global_step": global_step, "best_metric": best_metric, "config": config, "rng": rng_state()}, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, restore_rng=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None: optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None: scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None: scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng"):
        state=payload["rng"]; random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None: torch.cuda.set_rng_state_all(state["cuda"])
    return payload
