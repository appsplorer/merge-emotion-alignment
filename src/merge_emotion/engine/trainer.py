from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import GradScaler

from merge_emotion.engine.checkpoint import save_checkpoint
from merge_emotion.engine.evaluator import evaluate, move_batch
from merge_emotion.objectives.affect_aware import affect_aware_contrastive_loss
from merge_emotion.objectives.contrastive import symmetric_contrastive_loss
from merge_emotion.objectives.emotion import emotion_loss
from merge_emotion.optim.cspa import CSPAController, SecantCurvatureEstimator
from merge_emotion.optim.gradients import assign_flat_grad, flatten_grads
from merge_emotion.optim.multitask import cagrad_two_task, pcgrad_two_task


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    def factor(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _autocast_context(device, precision):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", enabled=False)


def train(model, train_loader, val_loader, device, config: Dict[str, Any], run_dir: Path, checkpoint_dir: Path):
    modality = config["experiment"]["modality"]
    tcfg = config["training"]
    ocfg = config["optimizer"]
    ecfg = config["emotion_loss"]
    ccfg = config["contrastive"]
    acfg = config["affect_aware"]
    cspa_cfg = config["cspa"]
    strategy = config.get("gradient_strategy", {"name": "scalarization"})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ocfg["learning_rate"]), weight_decay=float(ocfg["weight_decay"]))
    total_steps = int(tcfg["max_epochs"]) * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, float(config["scheduler"]["warmup_ratio"]))
    scaler = GradScaler(enabled=(device.type == "cuda" and config["runtime"]["precision"] == "fp16"))
    cspa = CSPAController(cspa_cfg["lambda_max"], cspa_cfg["kappa"], cspa_cfg["rho"], cspa_cfg["ema_beta"], cspa_cfg["eps"]) if cspa_cfg.get("enabled") else None
    curvature = SecantCurvatureEstimator(cspa_cfg["curvature_safety_factor"], cspa_cfg["curvature_min"], cspa_cfg["curvature_max"], cspa_cfg["eps"]) if cspa else None
    shared_params = [p for p in model.representation_parameters() if p.requires_grad]
    all_params = [p for p in model.parameters() if p.requires_grad]
    history, grad_history = [], []
    best = -float("inf")
    patience = 0
    global_step = 0
    for epoch in range(int(tcfg["max_epochs"])):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, config["runtime"]["precision"]):
                out = model(batch, modality=modality)
                emo = emotion_loss(out, batch["label"], modality, float(ecfg["lambda_uni"]), float(ecfg["label_smoothing"]))
                align = None
                if modality == "multimodal" and ccfg.get("enabled"):
                    if acfg.get("enabled"):
                        align = affect_aware_contrastive_loss(out["audio_repr"], out["lyrics_repr"], batch["label"], float(ccfg["temperature"]), float(acfg["epsilon"]), float(acfg["gamma"]))
                    else:
                        align = symmetric_contrastive_loss(out["audio_repr"], out["lyrics_repr"], float(ccfg["temperature"]))
            manual_gradient = False
            if align is not None and strategy.get("name") in {"pcgrad", "cagrad"}:
                ge = torch.autograd.grad(emo, all_params, retain_graph=True, allow_unused=True)
                ga = torch.autograd.grad(align, all_params, retain_graph=True, allow_unused=True)
                ge_flat = flatten_grads(ge, all_params).detach()
                ga_flat = flatten_grads(ga, all_params).detach()
                if strategy["name"] == "pcgrad":
                    combined = pcgrad_two_task(ge_flat, ga_flat)
                else:
                    combined = cagrad_two_task(ge_flat, ga_flat, float(strategy.get("c", 0.5)))
                assign_flat_grad(all_params, combined)
                total = emo + align
                manual_gradient = True
            elif cspa is not None and align is not None:
                # Geometry is measured on shared representation parameters. AdamW is the practical optimizer;
                # the raw-gradient descent guarantee is therefore treated as diagnostic rather than certified.
                ge_shared = torch.autograd.grad(emo, shared_params, retain_graph=True, allow_unused=True)
                ga_shared = torch.autograd.grad(align, shared_params, retain_graph=True, allow_unused=True)
                ge_flat = flatten_grads(ge_shared, shared_params).detach()
                ga_flat = flatten_grads(ga_shared, shared_params).detach()
                theta_flat = torch.cat([p.detach().reshape(-1) for p in shared_params])
                beta = curvature.update(theta_flat, ge_flat)
                state = cspa.choose(ge_flat, ga_flat, optimizer.param_groups[0]["lr"], beta)
                total = emo + state.lambda_used * align
                grad_history.append({"step": global_step, **state.__dict__})
            else:
                weight = float(ccfg.get("lambda_align", 0.0)) if align is not None else 0.0
                total = emo + weight * align if align is not None else emo
            if manual_gradient:
                torch.nn.utils.clip_grad_norm_(all_params, float(ocfg["gradient_clip_norm"]))
                optimizer.step()
            else:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, float(ocfg["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            running += float(total.detach())
            global_step += 1
        val_metrics, _ = evaluate(model, val_loader, device, modality)
        record = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "lr": optimizer.param_groups[0]["lr"]}
        record.update({"val_" + k: v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        history.append(record)
        metric = float(val_metrics["macro_f1"])
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, global_step, best, config)
        if metric > best:
            best = metric
            patience = 0
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, global_step, best, config)
        else:
            patience += 1
            if patience >= int(tcfg["early_stopping_patience"]):
                break
    return history, grad_history, best
