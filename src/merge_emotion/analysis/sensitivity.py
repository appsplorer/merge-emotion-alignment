from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import pandas as pd


ARCHITECTURE_SEEDS = (42, 123, 777)


def architecture_variants() -> Dict[str, dict]:
    """
    One-factor-at-a-time architecture sensitivity around the frozen
    reference architecture.

    Reference:
      hidden=256
      modality Transformer layers=2
      fusion Transformer layers=2
      heads=4
      FFN=512
    """
    return OrderedDict(
        [
            (
                "base",
                {
                    "hidden_dim": 256,
                    "modality_layers": 2,
                    "fusion_layers": 2,
                    "heads": 4,
                    "ffn_dim": 512,
                },
            ),
            (
                "hidden_128",
                {
                    "hidden_dim": 128,
                    "modality_layers": 2,
                    "fusion_layers": 2,
                    "heads": 4,
                    "ffn_dim": 256,
                },
            ),
            (
                "hidden_384",
                {
                    "hidden_dim": 384,
                    "modality_layers": 2,
                    "fusion_layers": 2,
                    "heads": 4,
                    "ffn_dim": 768,
                },
            ),
            (
                "encoder_1",
                {
                    "hidden_dim": 256,
                    "modality_layers": 1,
                    "fusion_layers": 2,
                    "heads": 4,
                    "ffn_dim": 512,
                },
            ),
            (
                "encoder_3",
                {
                    "hidden_dim": 256,
                    "modality_layers": 3,
                    "fusion_layers": 2,
                    "heads": 4,
                    "ffn_dim": 512,
                },
            ),
            (
                "fusion_1",
                {
                    "hidden_dim": 256,
                    "modality_layers": 2,
                    "fusion_layers": 1,
                    "heads": 4,
                    "ffn_dim": 512,
                },
            ),
            (
                "fusion_3",
                {
                    "hidden_dim": 256,
                    "modality_layers": 2,
                    "fusion_layers": 3,
                    "heads": 4,
                    "ffn_dim": 512,
                },
            ),
            (
                "heads_2",
                {
                    "hidden_dim": 256,
                    "modality_layers": 2,
                    "fusion_layers": 2,
                    "heads": 2,
                    "ffn_dim": 512,
                },
            ),
            (
                "heads_8",
                {
                    "hidden_dim": 256,
                    "modality_layers": 2,
                    "fusion_layers": 2,
                    "heads": 8,
                    "ffn_dim": 512,
                },
            ),
        ]
    )


def build_training_command(
    variant_name: str,
    spec: dict,
    seed: int,
    python_executable: str = "python",
) -> List[str]:
    run_id = "archsens_%s_seed%d" % (variant_name, seed)

    overrides = [
        "project.seed=%d" % seed,
        "experiment.name=archsens_%s" % variant_name,
        "experiment.stage=architecture_sensitivity",
        # Hold the already validation-selected alignment weight fixed.
        "contrastive.lambda_align=0.2",
        "audio_model.hidden_dim=%d" % spec["hidden_dim"],
        "lyrics_model.hidden_dim=%d" % spec["hidden_dim"],
        "multimodal.hidden_dim=%d" % spec["hidden_dim"],
        "audio_model.transformer_layers=%d" % spec["modality_layers"],
        "lyrics_model.transformer_layers=%d" % spec["modality_layers"],
        "multimodal.fusion_layers=%d" % spec["fusion_layers"],
        "audio_model.attention_heads=%d" % spec["heads"],
        "lyrics_model.attention_heads=%d" % spec["heads"],
        "multimodal.attention_heads=%d" % spec["heads"],
        "audio_model.ffn_dim=%d" % spec["ffn_dim"],
        "lyrics_model.ffn_dim=%d" % spec["ffn_dim"],
        "multimodal.ffn_dim=%d" % spec["ffn_dim"],
    ]

    cmd = [
        python_executable,
        "scripts/train.py",
        "--config",
        "configs/pipeline_experiments/multimodal_fixed.yaml",
        "--tuned",
        "results/manifests/optuna_best.yaml",
        "--run-id",
        run_id,
        "--resume",
        "auto",
        "--require-cuda",
    ]

    for item in overrides:
        cmd.extend(["--set", item])

    return cmd


def summarize_architecture_results(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"variant", "seed", "macro_f1", "accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Missing architecture columns: %s" % sorted(missing))

    metrics = ["macro_f1", "accuracy"]
    if "alignment_gap" in frame.columns:
        metrics.append("alignment_gap")

    grouped = frame.groupby("variant", sort=False)

    rows = []
    for variant, part in grouped:
        row = {
            "variant": variant,
            "n": int(len(part)),
        }
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce")
            row["%s_mean" % metric] = float(values.mean())
            row["%s_sd" % metric] = float(values.std(ddof=1))
            row["%s_min" % metric] = float(values.min())
            row["%s_max" % metric] = float(values.max())
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_component_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "multimodal_fixed",
        "affect_only",
        "cspa_only",
        "cspa_affect",
    ]

    data = frame[frame["experiment"].isin(methods)].copy()

    rows = []
    for method in methods:
        part = data[data["experiment"] == method]
        if part.empty:
            raise ValueError("Missing ablation experiment: %s" % method)

        row = {
            "experiment": method,
            "n": int(len(part)),
            "macro_f1_mean": float(part["macro_f1"].mean()),
            "macro_f1_sd": float(part["macro_f1"].std(ddof=1)),
            "macro_f1_min": float(part["macro_f1"].min()),
            "macro_f1_max": float(part["macro_f1"].max()),
        }

        if "accuracy" in part.columns:
            row["accuracy_mean"] = float(part["accuracy"].mean())
            row["accuracy_sd"] = float(part["accuracy"].std(ddof=1))

        if "alignment_gap" in part.columns:
            row["alignment_gap_mean"] = float(part["alignment_gap"].mean())
            row["alignment_gap_sd"] = float(part["alignment_gap"].std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_ablation_effects(frame: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "multimodal_fixed",
        "affect_only",
        "cspa_only",
        "cspa_affect",
    ]

    data = frame[frame["experiment"].isin(methods)][
        ["seed", "experiment", "macro_f1"]
    ].copy()

    pivot = data.pivot_table(
        index="seed",
        columns="experiment",
        values="macro_f1",
        aggfunc="first",
    ).dropna(subset=methods)

    if pivot.empty:
        raise ValueError("No complete paired seeds for ablation analysis")

    effects = OrderedDict(
        [
            (
                "affect_without_cspa",
                pivot["affect_only"] - pivot["multimodal_fixed"],
            ),
            (
                "cspa_without_affect",
                pivot["cspa_only"] - pivot["multimodal_fixed"],
            ),
            (
                "affect_given_cspa",
                pivot["cspa_affect"] - pivot["cspa_only"],
            ),
            (
                "cspa_given_affect",
                pivot["cspa_affect"] - pivot["affect_only"],
            ),
        ]
    )

    effects["interaction"] = (
        pivot["cspa_affect"]
        - pivot["cspa_only"]
        - pivot["affect_only"]
        + pivot["multimodal_fixed"]
    )

    rows = []
    for name, values in effects.items():
        rows.append(
            {
                "effect": name,
                "n_paired_seeds": int(len(values)),
                "delta_mean": float(values.mean()),
                "delta_sd": float(values.std(ddof=1)),
                "delta_min": float(values.min()),
                "delta_max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)
