#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

from merge_emotion.analysis.statistics import paired_seed_differences, seed_summary, spearman_relationship

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries"
OUT = ROOT / "results/tables"
OUT.mkdir(parents=True, exist_ok=True)

validation_path = SUMMARY / "all_runs.csv"
if validation_path.exists():
    validation = pd.read_csv(validation_path)
    if not validation.empty:
        seed_summary(validation).to_csv(OUT / "validation_seed_summary.csv", index=False)
        if "lambda_align" in validation.columns:
            sweep = validation[validation["experiment"].astype(str).str.startswith("lambda_")].copy()
            if not sweep.empty:
                seed_summary(sweep, group_col="lambda_align").to_csv(OUT / "lambda_sensitivity.csv", index=False)
                relation = spearman_relationship(sweep, "alignment_gap", "macro_f1")
                pd.DataFrame([relation]).to_csv(OUT / "alignment_vs_emotion_validation.csv", index=False)

final_path = SUMMARY / "final_test_runs.csv"
if final_path.exists():
    final = pd.read_csv(final_path)
    if not final.empty:
        seed_summary(final).to_csv(OUT / "final_test_seed_summary.csv", index=False)
        reference = "multimodal_fixed"
        if reference in set(final["experiment"].astype(str)):
            paired_seed_differences(final, reference=reference).to_csv(
                OUT / "paired_seed_differences_vs_multimodal_fixed.csv", index=False
            )
        if {"alignment_gap", "macro_f1"}.issubset(final.columns):
            relation = spearman_relationship(final, "alignment_gap", "macro_f1")
            pd.DataFrame([relation]).to_csv(OUT / "alignment_vs_emotion_test.csv", index=False)

print("Saved statistical tables to", OUT)
