#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from merge_emotion.analysis.statistics import paired_prediction_bootstrap_macro_f1

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final_test"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

records = []
for metrics_path in sorted(FINAL.glob("*/metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records.append(metrics)
frame = pd.DataFrame(records)
if frame.empty:
    raise SystemExit("No final-test metrics found")

reference = "multimodal_fixed"
rows = []
for experiment in sorted(frame["experiment"].unique()):
    if experiment == reference:
        continue
    for seed in sorted(set(frame[frame["experiment"] == experiment]["seed"])):
        ref_match = frame[(frame["experiment"] == reference) & (frame["seed"] == seed)]
        cmp_match = frame[(frame["experiment"] == experiment) & (frame["seed"] == seed)]
        if len(ref_match) != 1 or len(cmp_match) != 1:
            continue
        ref_id = ref_match.iloc[0]["run_id"]
        cmp_id = cmp_match.iloc[0]["run_id"]
        ref_pred = pd.read_csv(FINAL / ref_id / "predictions.csv")
        cmp_pred = pd.read_csv(FINAL / cmp_id / "predictions.csv")
        result = paired_prediction_bootstrap_macro_f1(ref_pred, cmp_pred, samples=5000, seed=2026 + int(seed))
        rows.append({"reference": reference, "comparison": experiment, "seed": int(seed), **result})

pd.DataFrame(rows).to_csv(OUT / "paired_prediction_bootstrap_vs_multimodal_fixed.csv", index=False)
print("Saved", OUT / "paired_prediction_bootstrap_vs_multimodal_fixed.csv")
