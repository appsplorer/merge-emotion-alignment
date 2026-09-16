from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _label_map() -> dict[str, str]:
    return {
        "audio_only": "Audio only",
        "lyrics_only": "Lyrics only",
        "multimodal_noalign": "No alignment",
        "multimodal_fixed": "Fixed alignment",
        "pcgrad": "PCGrad",
        "cagrad": "CAGrad",
        "cspa_only": "CSPA only",
        "affect_only": "Affect only",
        "cspa_affect": "CSPA + affect",
    }


def _method_order() -> list[str]:
    return [
        "audio_only",
        "lyrics_only",
        "multimodal_noalign",
        "multimodal_fixed",
        "pcgrad",
        "cagrad",
        "cspa_only",
        "affect_only",
        "cspa_affect",
    ]


def _pretty_name(name: str) -> str:
    return _label_map().get(str(name), str(name))


def _ordered_summary(frame: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    order = _method_order()
    grouped = frame.groupby("experiment")[metric].agg(["mean", "std", "count"]).reset_index()
    grouped["order"] = grouped["experiment"].apply(lambda x: order.index(x) if x in order else 999)
    grouped["pretty"] = grouped["experiment"].apply(_pretty_name)
    grouped = grouped.sort_values(["order", "pretty"]).reset_index(drop=True)
    return grouped


def plot_main_performance(frame: pd.DataFrame, path: Path) -> None:
    summary = _ordered_summary(frame, "macro_f1")
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.barh(summary["pretty"], summary["mean"], xerr=summary["std"].fillna(0))
    for i, row in summary.iterrows():
        ax.text(
            row["mean"] + 0.002,
            i,
            f'{row["mean"]:.4f} ± {0.0 if pd.isna(row["std"]) else row["std"]:.4f}',
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Held-out test Macro-F1")
    ax.set_ylabel("Method")
    ax.set_title("Final five-seed emotion-prediction performance")
    ax.set_xlim(left=max(0.0, float(summary["mean"].min()) - 0.05), right=min(1.0, float(summary["mean"].max()) + 0.08))
    _save(fig, path)


def plot_alignment_vs_f1(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["alignment_gap", "macro_f1"]).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    methods = data["experiment"].astype(str).unique().tolist()
    label_map = _label_map()
    for method in methods:
        part = data[data["experiment"] == method]
        ax.scatter(part["alignment_gap"], part["macro_f1"], label=label_map.get(method, method), alpha=0.85)
    ax.set_xlabel("Alignment gap (matched minus unmatched cosine)")
    ax.set_ylabel("Held-out test Macro-F1")
    ax.set_title("Alignment versus emotion prediction")
    ax.legend(fontsize=7, loc="best")
    _save(fig, path)


def plot_lambda_sensitivity(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["lambda_align", "macro_f1"]).copy()
    if data.empty:
        return
    summary = data.groupby("lambda_align")["macro_f1"].agg(["mean", "std", "count"]).reset_index().sort_values("lambda_align")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.errorbar(summary["lambda_align"], summary["mean"], yerr=summary["std"].fillna(0), marker="o", linewidth=1.5)
    for _, row in summary.iterrows():
        ax.text(float(row["lambda_align"]), float(row["mean"]) + 0.0015, f'{row["mean"]:.4f}', ha="center", fontsize=8)
    ax.set_xlabel("Alignment weight λ")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Validation sensitivity to alignment weight")
    ax.set_xscale("symlog", linthresh=0.01)
    _save(fig, path)


def plot_pareto(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["alignment_gap", "macro_f1"]).copy()
    if data.empty:
        return
    summary = data.groupby("experiment")[["alignment_gap", "macro_f1"]].mean().reset_index()
    summary["pretty"] = summary["experiment"].apply(_pretty_name)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(summary["alignment_gap"], summary["macro_f1"])
    for _, row in summary.iterrows():
        ax.annotate(row["pretty"], (row["alignment_gap"], row["macro_f1"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean alignment gap")
    ax.set_ylabel("Mean Macro-F1")
    ax.set_title("Method-level prediction–alignment trade-off")
    _save(fig, path)


def plot_training_curve(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    if "epoch" not in history.columns:
        history = history.reset_index().rename(columns={"index": "epoch"})
        history["epoch"] += 1
    if "train_loss" in history.columns:
        ax.plot(history["epoch"], history["train_loss"], label="train loss")
    if "val_macro_f1" in history.columns:
        ax.plot(history["epoch"], history["val_macro_f1"], label="validation Macro-F1")
    if "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Training dynamics")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_cspa_diagnostics(gradients: pd.DataFrame, path: Path) -> None:
    if gradients.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if "lambda_used" in gradients.columns:
        ax.plot(gradients["step"], gradients["lambda_used"], label="adaptive λ")
    if "grad_cosine" in gradients.columns:
        ax.plot(gradients["step"], gradients["grad_cosine"], label="gradient cosine")
    if "directional_margin" in gradients.columns:
        ax.plot(gradients["step"], gradients["directional_margin"], label="directional margin")
    if "relative_norm_cap" in gradients.columns:
        ax.plot(gradients["step"], gradients["relative_norm_cap"], label="relative norm cap")
    ax.set_xlabel("Training step")
    ax.set_title("CSPA optimization diagnostics")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_confusion(predictions: pd.DataFrame, path: Path, title: str = "Final held-out test confusion matrix") -> None:
    cm = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(5.1, 5.1))
    ConfusionMatrixDisplay(cm, display_labels=["Q1", "Q2", "Q3", "Q4"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    _save(fig, path)


def plot_method_alignment_summary(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["alignment_gap", "macro_f1"]).copy()
    if data.empty:
        return
    summary = data.groupby("experiment")[["alignment_gap", "macro_f1"]].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary["pretty"] = summary["experiment"].apply(_pretty_name)
    order = _method_order()
    summary["order"] = summary["experiment"].apply(lambda x: order.index(x) if x in order else 999)
    summary = summary.sort_values("order")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(
        summary["alignment_gap_mean"],
        summary["macro_f1_mean"],
        xerr=summary["alignment_gap_std"].fillna(0),
        yerr=summary["macro_f1_std"].fillna(0),
        fmt="o",
        capsize=3,
    )
    for _, row in summary.iterrows():
        ax.annotate(row["pretty"], (row["alignment_gap_mean"], row["macro_f1_mean"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean alignment gap ± SD")
    ax.set_ylabel("Mean Macro-F1 ± SD")
    ax.set_title("Method-level alignment–prediction summary")
    _save(fig, path)


def plot_per_quadrant_summary(frame: pd.DataFrame, path: Path, experiment: str = "cspa_affect") -> None:
    data = frame[frame["experiment"] == experiment].copy()
    needed = [
        "q1_precision", "q1_recall", "q1_f1",
        "q2_precision", "q2_recall", "q2_f1",
        "q3_precision", "q3_recall", "q3_f1",
        "q4_precision", "q4_recall", "q4_f1",
    ]
    if data.empty or any(col not in data.columns for col in needed):
        return

    rows = []
    for q in ("q1", "q2", "q3", "q4"):
        rows.append({
            "quadrant": q.upper(),
            "metric": "Precision",
            "mean": data[f"{q}_precision"].mean(),
            "std": data[f"{q}_precision"].std(),
        })
        rows.append({
            "quadrant": q.upper(),
            "metric": "Recall",
            "mean": data[f"{q}_recall"].mean(),
            "std": data[f"{q}_recall"].std(),
        })
        rows.append({
            "quadrant": q.upper(),
            "metric": "F1",
            "mean": data[f"{q}_f1"].mean(),
            "std": data[f"{q}_f1"].std(),
        })

    plot_df = pd.DataFrame(rows)
    metrics = ["Precision", "Recall", "F1"]
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    x = list(range(len(quadrants)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    offsets = [-width, 0.0, width]
    for metric, offset in zip(metrics, offsets):
        part = plot_df[plot_df["metric"] == metric].set_index("quadrant").loc[quadrants].reset_index()
        xpos = [xi + offset for xi in x]
        ax.bar(xpos, part["mean"], width=width, yerr=part["std"].fillna(0), label=metric, capsize=3)
        for px, val in zip(xpos, part["mean"]):
            ax.text(px, float(val) + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(quadrants)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Quadrant-wise performance for CSPA + affect (five held-out seeds)")
    ax.legend(fontsize=8)
    _save(fig, path)


def _draw_box(ax, x, y, w, h, text, fontsize=9):
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _line(ax, points: Iterable[tuple[float, float]], lw: float = 1.3):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, linewidth=lw)


def _arrowhead(ax, x0, y0, x1, y1, size=0.015):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.3, shrinkA=0, shrinkB=0))


def plot_architecture_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # left input boxes
    _draw_box(ax, 0.03, 0.73, 0.16, 0.10, "Audio waveform")
    _draw_box(ax, 0.03, 0.22, 0.16, 0.10, "Lyrics text")

    # frozen backbones
    _draw_box(ax, 0.24, 0.70, 0.18, 0.16, "Frozen MERT\nframe/chunk encoder")
    _draw_box(ax, 0.24, 0.19, 0.18, 0.16, "Frozen RoBERTa\nchunk encoder")

    # cached reps
    _draw_box(ax, 0.46, 0.72, 0.14, 0.12, "6 audio\nrepresentations")
    _draw_box(ax, 0.46, 0.21, 0.14, 0.12, "8 lyric\nrepresentations")

    # trainable modality transformers
    _draw_box(ax, 0.64, 0.70, 0.16, 0.16, "Trainable audio\nTransformer")
    _draw_box(ax, 0.64, 0.19, 0.16, 0.16, "Trainable lyrics\nTransformer")

    # fusion/cross-attn
    _draw_box(ax, 0.82, 0.50, 0.15, 0.20, "Bidirectional gated\ncross-attention\n+\nfusion Transformer")

    # heads/objectives
    _draw_box(ax, 0.82, 0.80, 0.15, 0.10, "Emotion head\n(Q1–Q4)")
    _draw_box(ax, 0.82, 0.31, 0.15, 0.10, "Alignment space\n(InfoNCE / affect)")

    # lines top branch
    _line(ax, [(0.19, 0.78), (0.24, 0.78)])
    _arrowhead(ax, 0.23, 0.78, 0.24, 0.78)

    _line(ax, [(0.42, 0.78), (0.46, 0.78)])
    _arrowhead(ax, 0.45, 0.78, 0.46, 0.78)

    _line(ax, [(0.60, 0.78), (0.64, 0.78)])
    _arrowhead(ax, 0.63, 0.78, 0.64, 0.78)

    # lines bottom branch
    _line(ax, [(0.19, 0.27), (0.24, 0.27)])
    _arrowhead(ax, 0.23, 0.27, 0.24, 0.27)

    _line(ax, [(0.42, 0.27), (0.46, 0.27)])
    _arrowhead(ax, 0.45, 0.27, 0.46, 0.27)

    _line(ax, [(0.60, 0.27), (0.64, 0.27)])
    _arrowhead(ax, 0.63, 0.27, 0.64, 0.27)

    # orthogonal merge into fusion
    _line(ax, [(0.80, 0.78), (0.83, 0.78), (0.83, 0.60)])
    _arrowhead(ax, 0.83, 0.61, 0.83, 0.60)

    _line(ax, [(0.80, 0.27), (0.83, 0.27), (0.83, 0.60)])
    _arrowhead(ax, 0.83, 0.59, 0.83, 0.60)

    # fusion to heads
    _line(ax, [(0.97, 0.60), (0.99, 0.60), (0.99, 0.85), (0.97, 0.85)])
    _arrowhead(ax, 0.98, 0.85, 0.97, 0.85)

    _line(ax, [(0.97, 0.60), (0.99, 0.60), (0.99, 0.36), (0.97, 0.36)])
    _arrowhead(ax, 0.98, 0.36, 0.97, 0.36)

    ax.text(0.50, 0.95, "Code-generated architecture used in the MERGE multimodal emotion pipeline", ha="center", va="center", fontsize=12)
    ax.text(0.50, 0.05, "Frozen backbone features -> trainable modality encoders -> gated bidirectional cross-attention -> fusion -> emotion/alignment objectives", ha="center", va="center", fontsize=9)
    _save(fig, path)
