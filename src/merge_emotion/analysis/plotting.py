from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_main_performance(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.groupby("experiment")["macro_f1"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.45 * len(summary))))
    ax.barh(summary.index, summary["mean"], xerr=summary["std"].fillna(0))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Emotion prediction performance")
    _save(fig, path)


def plot_alignment_vs_f1(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["alignment_gap", "macro_f1"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(data["alignment_gap"], data["macro_f1"])
    ax.set_xlabel("Matched-minus-unmatched cosine similarity")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Alignment versus emotion prediction")
    _save(fig, path)


def plot_lambda_sensitivity(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["lambda_align", "macro_f1"])
    if data.empty:
        return
    data = data.groupby("lambda_align")["macro_f1"].agg(["mean", "std"]).reset_index().sort_values("lambda_align")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.errorbar(data["lambda_align"], data["mean"], yerr=data["std"].fillna(0), marker="o")
    ax.set_xlabel("Alignment weight")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Alignment-strength sensitivity")
    _save(fig, path)


def plot_pareto(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["alignment_gap", "macro_f1"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(data["alignment_gap"], data["macro_f1"])
    for _, row in data.iterrows():
        ax.annotate(str(row["experiment"]), (row["alignment_gap"], row["macro_f1"]), fontsize=7)
    ax.set_xlabel("Alignment gap")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Prediction-alignment trade-off")
    _save(fig, path)


def plot_training_curve(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    if "val_macro_f1" in history:
        ax.plot(history["epoch"], history["val_macro_f1"], label="validation macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_title("Training behaviour")
    ax.legend()
    _save(fig, path)


def plot_cspa_diagnostics(gradients: pd.DataFrame, path: Path) -> None:
    if gradients.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(gradients["step"], gradients["lambda_used"], label="adaptive lambda")
    ax.plot(gradients["step"], gradients["grad_cosine"], label="gradient cosine")
    if "directional_margin" in gradients:
        ax.plot(gradients["step"], gradients["directional_margin"], label="directional margin")
    ax.set_xlabel("Training step")
    ax.set_title("CSPA optimization diagnostics")
    ax.legend()
    _save(fig, path)


def plot_confusion(predictions: pd.DataFrame, path: Path, title="Final held-out test confusion matrix") -> None:
    cm = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ConfusionMatrixDisplay(cm, display_labels=["Q1", "Q2", "Q3", "Q4"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    _save(fig, path)
