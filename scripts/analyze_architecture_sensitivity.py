#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from merge_emotion.analysis.sensitivity import (
    ARCHITECTURE_SEEDS,
    architecture_variants,
    summarize_ablation_effects,
    summarize_architecture_results,
    summarize_component_ablation,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "runs"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "figures"
MANIFEST_DIR = ROOT / "results" / "manifests"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def collect_architecture_results() -> pd.DataFrame:
    rows = []

    variants = architecture_variants()

    for variant, spec in variants.items():
        for seed in ARCHITECTURE_SEEDS:
            run_id = "archsens_%s_seed%d" % (variant, seed)
            metrics_path = RUN_ROOT / run_id / "metrics.json"

            if not metrics_path.exists():
                raise FileNotFoundError(
                    "Missing completed architecture metric: %s" % metrics_path
                )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            if metrics.get("split") != "validation":
                raise RuntimeError(
                    "%s is not a validation result: split=%r"
                    % (run_id, metrics.get("split"))
                )

            if metrics.get("test_evaluated") is not False:
                raise RuntimeError(
                    "%s does not satisfy test firewall: test_evaluated=%r"
                    % (run_id, metrics.get("test_evaluated"))
                )

            row = {
                "variant": variant,
                "seed": int(seed),
                "run_id": run_id,
                "hidden_dim": int(spec["hidden_dim"]),
                "modality_layers": int(spec["modality_layers"]),
                "fusion_layers": int(spec["fusion_layers"]),
                "heads": int(spec["heads"]),
                "ffn_dim": int(spec["ffn_dim"]),
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "split": metrics["split"],
                "test_evaluated": metrics["test_evaluated"],
            }

            if metrics.get("alignment_gap") is not None:
                row["alignment_gap"] = float(metrics["alignment_gap"])

            rows.append(row)

    frame = pd.DataFrame(rows)

    expected = len(variants) * len(ARCHITECTURE_SEEDS)
    if len(frame) != expected:
        raise RuntimeError(
            "Expected %d architecture results, found %d"
            % (expected, len(frame))
        )

    return frame


def paired_vs_base(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index="seed",
        columns="variant",
        values="macro_f1",
        aggfunc="first",
    )

    rows = []

    for variant in architecture_variants():
        if variant == "base":
            continue

        if variant not in pivot:
            raise ValueError("Missing architecture variant: %s" % variant)

        paired = pivot[variant] - pivot["base"]

        rows.append(
            {
                "variant": variant,
                "n_paired_seeds": int(paired.notna().sum()),
                "delta_macro_f1_mean": float(paired.mean()),
                "delta_macro_f1_sd": float(paired.std(ddof=1)),
                "delta_macro_f1_min": float(paired.min()),
                "delta_macro_f1_max": float(paired.max()),
            }
        )

    return pd.DataFrame(rows)


def plot_architecture(summary: pd.DataFrame, output: Path) -> None:
    order = list(architecture_variants())

    data = (
        summary.set_index("variant")
        .loc[order]
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    x = range(len(data))

    ax.errorbar(
        list(x),
        data["macro_f1_mean"],
        yerr=data["macro_f1_sd"].fillna(0),
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )

    for i, row in data.iterrows():
        ax.text(
            i,
            row["macro_f1_mean"] + 0.002,
            "%.4f" % row["macro_f1_mean"],
            ha="center",
            va="bottom",
            fontsize=8,
        )

    base_value = float(
        data.loc[data["variant"] == "base", "macro_f1_mean"].iloc[0]
    )

    ax.axhline(
        base_value,
        linestyle="--",
        linewidth=1.0,
        label="Reference architecture",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(data["variant"], rotation=35, ha="right")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_xlabel("Architecture variant")
    ax.set_title(
        "Architecture sensitivity on validation data "
        "(3 seeds; mean ± SD)"
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(table: pd.DataFrame, output: Path) -> None:
    order = [
        "multimodal_fixed",
        "affect_only",
        "cspa_only",
        "cspa_affect",
    ]

    labels = {
        "multimodal_fixed": "Fixed",
        "affect_only": "Affect only",
        "cspa_only": "CSPA only",
        "cspa_affect": "CSPA + affect",
    }

    data = table.set_index("experiment").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    x = range(len(data))

    ax.bar(
        list(x),
        data["macro_f1_mean"],
        yerr=data["macro_f1_sd"],
        capsize=4,
    )

    for i, row in data.iterrows():
        ax.text(
            i,
            row["macro_f1_mean"] + 0.002,
            "%.4f" % row["macro_f1_mean"],
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [labels[x] for x in data["experiment"]],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Held-out test Macro-F1")
    ax.set_title(
        "Component ablation "
        "(5 frozen test seeds; mean ± SD)"
    )

    low = max(
        0.0,
        float(data["macro_f1_mean"].min()) - 0.05,
    )
    high = min(
        1.0,
        float(data["macro_f1_mean"].max()) + 0.04,
    )
    ax.set_ylim(low, high)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    arch = collect_architecture_results()

    raw_arch_path = (
        TABLE_DIR / "architecture_sensitivity_validation.csv"
    )
    arch.to_csv(raw_arch_path, index=False)

    summary = summarize_architecture_results(arch)

    spec_frame = pd.DataFrame(
        [
            {"variant": variant, **spec}
            for variant, spec in architecture_variants().items()
        ]
    )

    summary = spec_frame.merge(
        summary,
        on="variant",
        how="left",
        validate="one_to_one",
    )

    summary_path = (
        TABLE_DIR / "architecture_sensitivity_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    paired = paired_vs_base(arch)
    paired_path = (
        TABLE_DIR / "architecture_sensitivity_paired_vs_base.csv"
    )
    paired.to_csv(paired_path, index=False)

    final_path = (
        ROOT / "results" / "summaries" / "final_test_runs.csv"
    )

    if not final_path.exists():
        raise FileNotFoundError(final_path)

    final = pd.read_csv(final_path)

    ablation = summarize_component_ablation(final)
    ablation_path = (
        TABLE_DIR / "component_ablation_final_test.csv"
    )
    ablation.to_csv(ablation_path, index=False)

    effects = summarize_ablation_effects(final)
    effects_path = (
        TABLE_DIR / "component_ablation_paired_effects.csv"
    )
    effects.to_csv(effects_path, index=False)

    arch_figures = []
    ablation_figures = []

    for ext in ("pdf", "png"):
        arch_out = (
            FIG_DIR / ("architecture_sensitivity_validation." + ext)
        )
        plot_architecture(summary, arch_out)
        arch_figures.append(arch_out)

        ablation_out = (
            FIG_DIR / ("component_ablation_final_test." + ext)
        )
        plot_ablation(ablation, ablation_out)
        ablation_figures.append(ablation_out)

    artifacts = [
        raw_arch_path,
        summary_path,
        paired_path,
        ablation_path,
        effects_path,
        *arch_figures,
        *ablation_figures,
    ]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "protocol": {
            "architecture_split": "validation",
            "architecture_test_evaluated": False,
            "architecture_seeds": list(ARCHITECTURE_SEEDS),
            "architecture_variants": architecture_variants(),
            "lambda_align_fixed": 0.2,
            "architecture_selection_use": (
                "diagnostic robustness analysis only; "
                "does not revise frozen held-out test results"
            ),
            "ablation_source": (
                "existing frozen five-seed held-out final_test_runs.csv"
            ),
        },
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }

    manifest_path = (
        MANIFEST_DIR / "architecture_sensitivity_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("ARCHITECTURE SENSITIVITY SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))

    print()
    print("=" * 78)
    print("PAIRED DIFFERENCE VS REFERENCE")
    print("=" * 78)
    print(paired.to_string(index=False))

    print()
    print("=" * 78)
    print("COMPONENT ABLATION")
    print("=" * 78)
    print(ablation.to_string(index=False))

    print()
    print("=" * 78)
    print("PAIRED ABLATION EFFECTS")
    print("=" * 78)
    print(effects.to_string(index=False))

    print()
    print("Manifest:", manifest_path)
    print("All analyses completed without test-set reuse for architecture sensitivity.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
