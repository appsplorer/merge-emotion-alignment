#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Freeze validation-selected final runs before opening test.")
    p.add_argument("--summary", default="results/summaries/all_runs.csv")
    p.add_argument("--output", default="results/manifests/final_selection.yaml")
    p.add_argument("--stage", default="final_seed")
    args = p.parse_args()

    frame = pd.read_csv(PROJECT_ROOT / args.summary)
    selected = frame[frame["stage"] == args.stage].copy()
    if selected.empty:
        raise ValueError("No final_seed validation runs found")
    if "test_evaluated" in selected.columns and selected["test_evaluated"].fillna(False).astype(bool).any():
        raise ValueError("A selected training run indicates test exposure")
    selected = selected.sort_values(["experiment", "seed", "run_id"])
    duplicates = selected.duplicated(subset=["experiment", "seed"], keep=False)
    if duplicates.any():
        raise ValueError("Multiple final runs exist for the same experiment/seed; resolve before freezing")

    runs = []
    for _, row in selected.iterrows():
        run_id = str(row["run_id"])
        run_dir = PROJECT_ROOT / "results" / "runs" / run_id
        checkpoint = PROJECT_ROOT / "checkpoints" / run_id / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        runs.append(
            {
                "run_id": run_id,
                "experiment": str(row["experiment"]),
                "seed": int(row["seed"]),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "selection_metric": "validation_macro_f1",
                "validation_macro_f1": float(row["macro_f1"]),
            }
        )

    payload = {
        "frozen": True,
        "selection_source": "validation_only",
        "stage": args.stage,
        "n_runs": len(runs),
        "runs": runs,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print("Frozen", len(runs), "runs ->", output)


if __name__ == "__main__":
    main()
