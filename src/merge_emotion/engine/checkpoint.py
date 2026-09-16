from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def config_fingerprint(config: Dict[str, Any]) -> str:
    """Stable SHA-256 fingerprint used to reject incompatible training resumes."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: Dict[str, Any],
    trainer_state: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_metric": float(best_metric),
            "config": config,
            "config_fingerprint": config_fingerprint(config),
            "trainer_state": trainer_state or {},
            "rng_state": rng_state(),
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, restore_rng=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        state = payload.get("rng_state", payload.get("rng"))
        if state is not None:
            restore_rng_state(state)
    return payload
