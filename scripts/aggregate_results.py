#!/usr/bin/env python3
from pathlib import Path

import json
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def collect_validation_runs():
    rows = []
    for metrics_path in sorted((ROOT / "results/runs").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = metrics_path.parent / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        row = dict(metrics)
        row.setdefault("run_id", metrics_path.parent.name)
        row["lambda_align"] = config.get("contrastive", {}).get("lambda_align")
        row["temperature"] = config.get("contrastive", {}).get("temperature")
        row["lambda_uni"] = config.get("emotion_loss", {}).get("lambda_uni")
        row["gradient_strategy"] = config.get("gradient_strategy", {}).get("name", "scalarization")
        row["affect_aware"] = bool(config.get("affect_aware", {}).get("enabled", False))
        row["cspa"] = bool(config.get("cspa", {}).get("enabled", False))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_final_test_runs():
    rows = []
    for metrics_path in sorted((ROOT / "results/final_test").glob("*/metrics.json")):
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(row)
    return pd.DataFrame(rows)


out = ROOT / "results/summaries"
out.mkdir(parents=True, exist_ok=True)
validation = collect_validation_runs()
validation.to_csv(out / "all_runs.csv", index=False)
final_test = collect_final_test_runs()
final_test.to_csv(out / "final_test_runs.csv", index=False)
print("Validation runs:", len(validation))
print("Final-test runs:", len(final_test))
