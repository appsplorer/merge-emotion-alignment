#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

from merge_emotion.analysis.plotting import (
    plot_alignment_vs_f1,
    plot_confusion,
    plot_cspa_diagnostics,
    plot_lambda_sensitivity,
    plot_main_performance,
    plot_pareto,
    plot_training_curve,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
validation_path = ROOT / "results/summaries/all_runs.csv"
final_path = ROOT / "results/summaries/final_test_runs.csv"

if validation_path.exists():
    validation = pd.read_csv(validation_path)
    if not validation.empty:
        sweep = validation[validation["experiment"].astype(str).str.startswith("lambda_")].copy()
        for ext in ("pdf", "png"):
            if not sweep.empty:
                plot_lambda_sensitivity(sweep, FIG / ("lambda_sensitivity_validation." + ext))
                plot_alignment_vs_f1(sweep, FIG / ("alignment_vs_f1_validation." + ext))
            cspa_rows = validation[validation["experiment"] == "cspa_affect"]
            if not cspa_rows.empty:
                selected = cspa_rows.sort_values("macro_f1", ascending=False).iloc[0]
                run_dir = ROOT / "results/runs" / selected["run_id"]
                if (run_dir / "history.csv").exists():
                    plot_training_curve(pd.read_csv(run_dir / "history.csv"), FIG / ("training_curve_cspa_validation." + ext))
                if (run_dir / "gradient_stats.csv").exists():
                    plot_cspa_diagnostics(pd.read_csv(run_dir / "gradient_stats.csv"), FIG / ("cspa_diagnostics_validation." + ext))

if final_path.exists():
    final = pd.read_csv(final_path)
    if not final.empty:
        for ext in ("pdf", "png"):
            plot_main_performance(final, FIG / ("main_performance_final_test." + ext))
            if {"alignment_gap", "macro_f1"}.issubset(final.columns):
                plot_alignment_vs_f1(final, FIG / ("alignment_vs_f1_final_test." + ext))
                plot_pareto(final, FIG / ("pareto_final_test." + ext))
        chosen = final[(final["experiment"] == "cspa_affect") & (final["seed"] == 42)]
        if not chosen.empty:
            run_id = chosen.iloc[0]["run_id"]
            pred_path = ROOT / "results/final_test" / run_id / "predictions.csv"
            if pred_path.exists():
                for ext in ("pdf", "png"):
                    plot_confusion(pd.read_csv(pred_path), FIG / ("confusion_cspa_affect_seed42." + ext))

print("Figures generated in", FIG)
