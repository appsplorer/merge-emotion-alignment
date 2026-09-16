#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from merge_emotion.analysis.plotting import (
    plot_alignment_vs_f1,
    plot_architecture_diagram,
    plot_confusion,
    plot_cspa_diagnostics,
    plot_lambda_sensitivity,
    plot_main_performance,
    plot_method_alignment_summary,
    plot_pareto,
    plot_per_quadrant_summary,
    plot_training_curve,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = ROOT / "results" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_PATH = ROOT / "results" / "summaries" / "all_runs.csv"
FINAL_PATH = ROOT / "results" / "summaries" / "final_test_runs.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def add_record(records: list[dict], name: str, inputs: list[Path], outputs: list[Path]) -> None:
    output_records = []
    for out in outputs:
        output_records.append({
            "path": str(out.relative_to(ROOT)),
            "exists": out.exists(),
            "size_bytes": out.stat().st_size if out.exists() else None,
            "sha256": sha256_file(out) if out.exists() else None,
        })
    records.append({
        "figure": name,
        "inputs": [str(p.relative_to(ROOT)) if p.exists() else str(p) for p in inputs],
        "outputs": output_records,
    })


def existing(path: Path) -> bool:
    return path.exists() and path.is_file()


def main() -> int:
    records: list[dict] = []
    validation = pd.read_csv(VALIDATION_PATH) if VALIDATION_PATH.exists() else pd.DataFrame()
    final = pd.read_csv(FINAL_PATH) if FINAL_PATH.exists() else pd.DataFrame()

    # 1. architecture diagram
    arch_outputs = []
    for ext in ("pdf", "png"):
        out = FIG / f"architecture_pipeline.{ext}"
        plot_architecture_diagram(out)
        arch_outputs.append(out)
    add_record(records, "architecture_pipeline", [], arch_outputs)

    # 2. validation sweep figures
    if not validation.empty:
        sweep = validation[validation["experiment"].astype(str).str.startswith("lambda_")].copy()
        if not sweep.empty:
            outs = []
            for ext in ("pdf", "png"):
                out = FIG / f"lambda_sensitivity_validation.{ext}"
                plot_lambda_sensitivity(sweep, out)
                outs.append(out)
            add_record(records, "lambda_sensitivity_validation", [VALIDATION_PATH], outs)

        cspa_rows = validation[validation["experiment"] == "cspa_affect"].copy()
        if not cspa_rows.empty:
            selected = cspa_rows.sort_values("macro_f1", ascending=False).iloc[0]
            run_dir = ROOT / "results" / "runs" / str(selected["run_id"])

            if (run_dir / "history.csv").exists():
                outs = []
                for ext in ("pdf", "png"):
                    out = FIG / f"training_curve_cspa_validation.{ext}"
                    plot_training_curve(pd.read_csv(run_dir / "history.csv"), out)
                    outs.append(out)
                add_record(records, "training_curve_cspa_validation", [run_dir / "history.csv"], outs)

            if (run_dir / "gradient_stats.csv").exists():
                outs = []
                for ext in ("pdf", "png"):
                    out = FIG / f"cspa_diagnostics_validation.{ext}"
                    plot_cspa_diagnostics(pd.read_csv(run_dir / "gradient_stats.csv"), out)
                    outs.append(out)
                add_record(records, "cspa_diagnostics_validation", [run_dir / "gradient_stats.csv"], outs)

    # 3. final test figures
    if not final.empty:
        outs = []
        for ext in ("pdf", "png"):
            out = FIG / f"main_performance_final_test.{ext}"
            plot_main_performance(final, out)
            outs.append(out)
        add_record(records, "main_performance_final_test", [FINAL_PATH], outs)

        if {"alignment_gap", "macro_f1"}.issubset(final.columns):
            outs = []
            for ext in ("pdf", "png"):
                out = FIG / f"alignment_vs_f1_final_test.{ext}"
                plot_alignment_vs_f1(final, out)
                outs.append(out)
            add_record(records, "alignment_vs_f1_final_test", [FINAL_PATH], outs)

            outs = []
            for ext in ("pdf", "png"):
                out = FIG / f"pareto_final_test.{ext}"
                plot_pareto(final, out)
                outs.append(out)
            add_record(records, "pareto_final_test", [FINAL_PATH], outs)

            outs = []
            for ext in ("pdf", "png"):
                out = FIG / f"method_alignment_summary_final_test.{ext}"
                plot_method_alignment_summary(final, out)
                outs.append(out)
            add_record(records, "method_alignment_summary_final_test", [FINAL_PATH], outs)

        outs = []
        for ext in ("pdf", "png"):
            out = FIG / f"per_quadrant_cspa_affect_final_test.{ext}"
            plot_per_quadrant_summary(final, out, experiment="cspa_affect")
            outs.append(out)
        add_record(records, "per_quadrant_cspa_affect_final_test", [FINAL_PATH], outs)

        chosen = final[(final["experiment"] == "cspa_affect") & (final["seed"] == 42)]
        if not chosen.empty:
            run_id = str(chosen.iloc[0]["run_id"])
            pred_path = ROOT / "results" / "final_test" / run_id / "predictions.csv"
            if pred_path.exists():
                outs = []
                for ext in ("pdf", "png"):
                    out = FIG / f"confusion_cspa_affect_seed42.{ext}"
                    plot_confusion(pd.read_csv(pred_path), out)
                    outs.append(out)
                add_record(records, "confusion_cspa_affect_seed42", [pred_path], outs)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "figures_dir": str(FIG.relative_to(ROOT)),
        "records": records,
    }
    manifest_path = FIG / "report_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Figures generated in", FIG)
    print("Manifest:", manifest_path)
    print("Generated figure records:", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
