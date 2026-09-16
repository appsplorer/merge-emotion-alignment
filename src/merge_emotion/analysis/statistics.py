from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def _bootstrap_mean_ci(values, samples=5000, seed=2026, alpha=0.05):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(int(samples), x.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, alpha / 2.0)), float(np.quantile(draws, 1.0 - alpha / 2.0))


def seed_summary(
    frame: pd.DataFrame,
    group_col: str = "experiment",
    metrics: Optional[Iterable[str]] = None,
    bootstrap_samples: int = 5000,
    seed: int = 2026,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if metrics is None:
        metrics = [
            c
            for c in (
                "macro_f1",
                "accuracy",
                "weighted_f1",
                "alignment_gap",
                "retrieval_a2l_r1",
                "retrieval_a2l_r5",
                "retrieval_l2a_r1",
                "retrieval_l2a_r5",
            )
            if c in frame.columns
        ]
    rows: List[Dict[str, float]] = []
    for group, block in frame.groupby(group_col, dropna=False):
        row: Dict[str, float] = {group_col: group, "n": int(len(block))}
        for metric in metrics:
            if metric not in block:
                continue
            values = pd.to_numeric(block[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            low, high = _bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=seed + sum(ord(ch) for ch in str(metric)),
            )
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[metric + "_ci_low"] = low
            row[metric + "_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def paired_seed_differences(
    frame: pd.DataFrame,
    reference: str,
    metric: str = "macro_f1",
    experiment_col: str = "experiment",
    seed_col: str = "seed",
    bootstrap_samples: int = 5000,
    rng_seed: int = 2026,
) -> pd.DataFrame:
    ref = frame[frame[experiment_col] == reference][[seed_col, metric]].rename(columns={metric: "reference"})
    rows = []
    for experiment in sorted(frame[experiment_col].dropna().unique()):
        if experiment == reference:
            continue
        other = frame[frame[experiment_col] == experiment][[seed_col, metric]].rename(columns={metric: "comparison"})
        paired = ref.merge(other, on=seed_col, how="inner")
        if paired.empty:
            continue
        diffs = paired["comparison"].to_numpy(float) - paired["reference"].to_numpy(float)
        low, high = _bootstrap_mean_ci(diffs, samples=bootstrap_samples, seed=rng_seed)
        rows.append(
            {
                "reference": reference,
                "comparison": experiment,
                "metric": metric,
                "n_pairs": len(diffs),
                "mean_difference": float(np.mean(diffs)),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def spearman_relationship(frame: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    data = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        return {"x": x, "y": y, "n": int(len(data)), "spearman_rho": np.nan, "p_value": np.nan}
    result = spearmanr(data[x].to_numpy(), data[y].to_numpy())
    return {
        "x": x,
        "y": y,
        "n": int(len(data)),
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def paired_prediction_bootstrap_macro_f1(
    reference_predictions: pd.DataFrame,
    comparison_predictions: pd.DataFrame,
    samples: int = 5000,
    seed: int = 2026,
) -> Dict[str, float]:
    required = {"song_id", "y_true", "y_pred"}
    if not required.issubset(reference_predictions.columns) or not required.issubset(comparison_predictions.columns):
        raise ValueError("Prediction frames require song_id, y_true, y_pred")
    merged = reference_predictions[list(required)].merge(
        comparison_predictions[list(required)],
        on="song_id",
        suffixes=("_ref", "_cmp"),
        validate="one_to_one",
    )
    if not np.array_equal(merged["y_true_ref"].to_numpy(), merged["y_true_cmp"].to_numpy()):
        raise ValueError("Paired predictions disagree on y_true")
    y = merged["y_true_ref"].to_numpy()
    ref = merged["y_pred_ref"].to_numpy()
    cmp = merged["y_pred_cmp"].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(samples), dtype=float)
    for i in range(int(samples)):
        idx = rng.integers(0, len(y), size=len(y))
        diffs[i] = f1_score(y[idx], cmp[idx], average="macro", zero_division=0) - f1_score(
            y[idx], ref[idx], average="macro", zero_division=0
        )
    return {
        "n": int(len(y)),
        "mean_difference": float(np.mean(diffs)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "probability_difference_gt_zero": float(np.mean(diffs > 0.0)),
    }
