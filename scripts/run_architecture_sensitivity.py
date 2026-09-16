#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from merge_emotion.analysis.sensitivity import (
    ARCHITECTURE_SEEDS,
    architecture_variants,
    build_training_command,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "runs"
CHECKPOINT_ROOT = ROOT / "checkpoints"
RECOVERY_ROOT = ROOT / "results" / "architecture_sensitivity" / "recovery"


def completed(run_id: str) -> bool:
    status = RUN_ROOT / run_id / "status.json"
    metrics = RUN_ROOT / run_id / "metrics.json"

    if not status.exists() or not metrics.exists():
        return False

    try:
        status_obj = json.loads(status.read_text(encoding="utf-8"))
        metrics_obj = json.loads(metrics.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        status_obj.get("status") == "completed"
        and metrics_obj.get("split") == "validation"
        and metrics_obj.get("test_evaluated") is False
    )


def archive_incomplete(run_id: str) -> None:
    run_dir = RUN_ROOT / run_id
    ckpt_dir = CHECKPOINT_ROOT / run_id
    last_ckpt = ckpt_dir / "last.pt"

    # A valid checkpoint can be resumed by train.py --resume auto.
    if last_ckpt.exists():
        return

    if not run_dir.exists() and not ckpt_dir.exists():
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = RECOVERY_ROOT / ("%s_%s" % (run_id, stamp))
    destination.mkdir(parents=True, exist_ok=True)

    if run_dir.exists():
        shutil.move(str(run_dir), str(destination / "run"))

    if ckpt_dir.exists():
        shutil.move(str(ckpt_dir), str(destination / "checkpoint"))


def main() -> int:
    os.chdir(ROOT)

    if not torch.cuda.is_available():
        raise RuntimeError("Architecture sensitivity requires CUDA")

    print("=" * 78)
    print("ARCHITECTURE SENSITIVITY")
    print("=" * 78)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Seeds:", ARCHITECTURE_SEEDS)
    print("Variants:", list(architecture_variants()))
    print("Total planned runs:", len(ARCHITECTURE_SEEDS) * len(architecture_variants()))
    print("IMPORTANT: train.py uses train/validation only; test is not evaluated.")
    print("=" * 78)

    failures = []

    for variant_name, spec in architecture_variants().items():
        for seed in ARCHITECTURE_SEEDS:
            run_id = "archsens_%s_seed%d" % (variant_name, seed)

            print()
            print("=" * 78)
            print("RUN:", run_id)
            print("SPEC:", spec)
            print("=" * 78)

            if completed(run_id):
                print("SKIP COMPLETED:", run_id)
                continue

            archive_incomplete(run_id)

            cmd = build_training_command(
                variant_name=variant_name,
                spec=spec,
                seed=seed,
                python_executable=sys.executable,
            )

            print("COMMAND:")
            print(" ".join(cmd), flush=True)

            try:
                subprocess.run(cmd, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as exc:
                failures.append((run_id, exc.returncode))
                print("FAILED:", run_id, "returncode=", exc.returncode, flush=True)
                # Fail immediately. A later Slurm resubmission can resume from last.pt.
                raise

            if not completed(run_id):
                raise RuntimeError(
                    "%s returned successfully but did not produce a valid "
                    "validation-only completed record" % run_id
                )

    if failures:
        print("Failures:", failures)
        return 1

    print()
    print("=" * 78)
    print("ALL 27 ARCHITECTURE-SENSITIVITY RUNS COMPLETED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
