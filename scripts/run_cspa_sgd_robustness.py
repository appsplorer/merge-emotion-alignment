#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from merge_emotion.analysis.small_extension import (
    build_theory_sgd_overrides,
)


ROOT = Path(__file__).resolve().parents[1]

SEEDS = (
    42,
    123,
    777,
)

RUN_ROOT = (
    ROOT
    / "results"
    / "runs"
)

CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints"
)

RECOVERY_ROOT = (
    ROOT
    / "results"
    / "extension_runs"
    / "recovery"
)


def is_complete(run_id: str) -> bool:
    run_dir = (
        RUN_ROOT
        / run_id
    )

    status_path = (
        run_dir
        / "status.json"
    )

    metrics_path = (
        run_dir
        / "metrics.json"
    )

    if not (
        status_path.exists()
        and metrics_path.exists()
    ):
        return False

    try:
        status = json.loads(
            status_path.read_text(
                encoding="utf-8"
            )
        )

        metrics = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return False

    return (
        status.get(
            "status"
        )
        == "completed"
        and metrics.get(
            "split"
        )
        == "validation"
        and metrics.get(
            "test_evaluated"
        )
        is False
    )


def prepare_resume(run_id: str):
    run_dir = (
        RUN_ROOT
        / run_id
    )

    checkpoint_dir = (
        CHECKPOINT_ROOT
        / run_id
    )

    last_checkpoint = (
        checkpoint_dir
        / "last.pt"
    )

    if (
        run_dir.exists()
        and last_checkpoint.exists()
    ):
        return

    if (
        not run_dir.exists()
        and not checkpoint_dir.exists()
    ):
        return

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    destination = (
        RECOVERY_ROOT
        / (
            "%s_%s"
            % (
                run_id,
                stamp,
            )
        )
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    if run_dir.exists():
        shutil.move(
            str(run_dir),
            str(
                destination
                / "run"
            ),
        )

    if checkpoint_dir.exists():
        shutil.move(
            str(checkpoint_dir),
            str(
                destination
                / "checkpoint"
            ),
        )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    print("=" * 78)
    print("CSPA RAW-SGD-STYLE OPTIMIZER-ASSUMPTION DIAGNOSTIC")
    print("=" * 78)

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "New training split: train/validation only"
    )

    print(
        "Held-out test: NOT evaluated"
    )

    print(
        "SGD momentum=0; weight_decay=0; clipping effectively disabled"
    )

    for seed in SEEDS:
        run_id = (
            "cspa_sgd_seed%d"
            % seed
        )

        print()
        print("=" * 78)
        print("RUN", run_id)
        print("=" * 78)

        if is_complete(
            run_id
        ):
            print(
                "Already completed; skipping."
            )

            continue

        prepare_resume(
            run_id
        )

        command = [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/pipeline_experiments/cspa_affect.yaml",
            "--tuned",
            "results/manifests/optuna_best.yaml",
            "--run-id",
            run_id,
            "--resume",
            "auto",
            "--require-cuda",
        ]

        for override in build_theory_sgd_overrides(
            seed
        ):
            command.extend(
                [
                    "--set",
                    override,
                ]
            )

        print(
            " ".join(
                command
            ),
            flush=True,
        )

        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )

        if not is_complete(
            run_id
        ):
            raise RuntimeError(
                "%s did not finish as a valid "
                "validation-only run"
                % run_id
            )

    print()
    print(
        "All three raw-SGD-style runs completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
