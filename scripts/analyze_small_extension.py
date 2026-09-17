#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merge_emotion.analysis.small_extension import (
    pca_tsne_projection,
)

from merge_emotion.metrics.classification import (
    classification_metrics,
)


ROOT = Path(__file__).resolve().parents[1]

REPRESENTATIONS = (
    ROOT
    / "results"
    / "representations"
)

TABLES = (
    ROOT
    / "results"
    / "tables"
)

FIGURES = (
    ROOT
    / "figures"
)

MANIFESTS = (
    ROOT
    / "results"
    / "manifests"
)

EXTENSION_RUNS = (
    ROOT
    / "results"
    / "extension_runs"
)

METHODS = [
    "multimodal_noalign",
    "multimodal_fixed",
    "cspa_affect",
]

METHOD_LABELS = {
    "multimodal_noalign":
        "No alignment",

    "multimodal_fixed":
        "Fixed alignment",

    "cspa_affect":
        "CSPA + affect",
}

SEEDS = [
    42,
    123,
    777,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
    ).strip()


def bool_series(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def load_representation(method: str):
    stem = (
        "%s_validation_seed42"
        % method
    )

    npz_path = (
        REPRESENTATIONS
        / (
            stem
            + ".npz"
        )
    )

    metadata_path = (
        REPRESENTATIONS
        / (
            stem
            + ".json"
        )
    )

    if not npz_path.exists():
        raise FileNotFoundError(
            npz_path
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        metadata["split"]
        != "validation"
        or metadata["test_used"]
        is not False
    ):
        raise RuntimeError(
            "Representation test firewall violation"
        )

    arrays = np.load(
        npz_path,
        allow_pickle=False,
    )

    return (
        npz_path,
        metadata_path,
        metadata,
        arrays,
    )


def plot_modality_projection(
    loaded,
    extension: str,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            13.0,
            4.5,
        ),
    )

    palette = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()[
        "color"
    ][
        :4
    ]

    marker_map = {
        "Audio": "o",
        "Lyrics": "^",
    }

    for axis, method in zip(
        axes,
        METHODS,
    ):
        _, _, _, arrays = loaded[
            method
        ]

        audio = arrays[
            "audio_repr"
        ]

        lyrics = arrays[
            "lyrics_repr"
        ]

        labels = arrays[
            "label"
        ]

        combined = np.concatenate(
            [
                audio,
                lyrics,
            ],
            axis=0,
        )

        projected = pca_tsne_projection(
            combined,
            random_state=42,
            perplexity=30,
        )

        n = len(
            labels
        )

        all_labels = np.concatenate(
            [
                labels,
                labels,
            ]
        )

        modalities = np.asarray(
            ["Audio"] * n
            + ["Lyrics"] * n
        )

        for q in range(4):
            for modality in (
                "Audio",
                "Lyrics",
            ):
                mask = (
                    (
                        all_labels
                        == q
                    )
                    &
                    (
                        modalities
                        == modality
                    )
                )

                axis.scatter(
                    projected[
                        mask,
                        0,
                    ],
                    projected[
                        mask,
                        1,
                    ],
                    s=12,
                    alpha=0.55,
                    marker=marker_map[
                        modality
                    ],
                    color=palette[
                        q
                    ],
                    label=(
                        "Q%d %s"
                        % (
                            q + 1,
                            modality,
                        )
                    ),
                )

        axis.set_title(
            METHOD_LABELS[
                method
            ]
        )

        axis.set_xlabel(
            "t-SNE dimension 1"
        )

        axis.set_ylabel(
            "t-SNE dimension 2"
        )

    handles, labels = (
        axes[-1]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=7,
        bbox_to_anchor=(
            0.5,
            -0.01,
        ),
    )

    fig.suptitle(
        "Audio--lyrics representation geometry on the validation split"
    )

    fig.tight_layout(
        rect=[
            0,
            0.08,
            1,
            0.95,
        ]
    )

    output = (
        FIGURES
        / (
            "representation_tsne_modalities_validation."
            + extension
        )
    )

    fig.savefig(
        output,
        bbox_inches="tight",
        dpi=220
        if extension
        == "png"
        else None,
    )

    plt.close(
        fig
    )

    return output


def plot_fused_projection(
    loaded,
    extension: str,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            12.6,
            4.4,
        ),
    )

    palette = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()[
        "color"
    ][
        :4
    ]

    for axis, method in zip(
        axes,
        METHODS,
    ):
        _, _, _, arrays = loaded[
            method
        ]

        embeddings = arrays[
            "multimodal_repr"
        ]

        labels = arrays[
            "label"
        ]

        projected = pca_tsne_projection(
            embeddings,
            random_state=42,
            perplexity=30,
        )

        for q in range(4):
            mask = (
                labels
                == q
            )

            axis.scatter(
                projected[
                    mask,
                    0,
                ],
                projected[
                    mask,
                    1,
                ],
                s=15,
                alpha=0.65,
                color=palette[
                    q
                ],
                label=(
                    "Q%d"
                    % (
                        q + 1
                    )
                ),
            )

        axis.set_title(
            METHOD_LABELS[
                method
            ]
        )

        axis.set_xlabel(
            "t-SNE dimension 1"
        )

        axis.set_ylabel(
            "t-SNE dimension 2"
        )

    handles, labels = (
        axes[-1]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(
            0.5,
            -0.01,
        ),
    )

    fig.suptitle(
        "Fused emotion representation on the validation split"
    )

    fig.tight_layout(
        rect=[
            0,
            0.07,
            1,
            0.95,
        ]
    )

    output = (
        FIGURES
        / (
            "representation_tsne_fused_validation."
            + extension
        )
    )

    fig.savefig(
        output,
        bbox_inches="tight",
        dpi=220
        if extension
        == "png"
        else None,
    )

    plt.close(
        fig
    )

    return output


def diagnostics_for_run(
    run_dir: Path,
):
    metrics_path = (
        run_dir
        / "metrics.json"
    )

    gradients_path = (
        run_dir
        / "gradient_stats.csv"
    )

    metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        metrics.get(
            "split"
        )
        != "validation"
        or metrics.get(
            "test_evaluated"
        )
        is not False
    ):
        raise RuntimeError(
            "Optimizer diagnostic test firewall failed: %s"
            % run_dir
        )

    gradients = pd.read_csv(
        gradients_path
    )

    result = {
        "macro_f1":
            float(
                metrics[
                    "macro_f1"
                ]
            ),

        "accuracy":
            float(
                metrics[
                    "accuracy"
                ]
            ),

        "alignment_gap":
            float(
                metrics.get(
                    "alignment_gap",
                    np.nan,
                )
            ),

        "n_gradient_steps":
            int(
                len(
                    gradients
                )
            ),
    }

    for column in (
        "lambda_used",
        "grad_cosine",
        "beta",
        "directional_margin",
    ):
        if column in gradients.columns:
            values = pd.to_numeric(
                gradients[
                    column
                ],
                errors="coerce",
            )

            result[
                column
                + "_mean"
            ] = float(
                values.mean()
            )

            result[
                column
                + "_min"
            ] = float(
                values.min()
            )

            result[
                column
                + "_max"
            ] = float(
                values.max()
            )

    if "constraint_active" in gradients.columns:
        result[
            "constraint_active_rate"
        ] = float(
            bool_series(
                gradients[
                    "constraint_active"
                ]
            ).mean()
        )

    if "feasible" in gradients.columns:
        result[
            "feasible_rate"
        ] = float(
            bool_series(
                gradients[
                    "feasible"
                ]
            ).mean()
        )

    if "directional_margin" in gradients.columns:
        margin = pd.to_numeric(
            gradients[
                "directional_margin"
            ],
            errors="coerce",
        )

        result[
            "negative_directional_margin_rate"
        ] = float(
            (
                margin
                < 0
            ).mean()
        )

    return result


def optimizer_diagnostic():
    rows = []

    for seed in SEEDS:
        runs = {
            "AdamW":
                (
                    ROOT
                    / "results"
                    / "runs"
                    / (
                        "final_cspa_affect_seed%d"
                        % seed
                    )
                ),

            "SGD_raw":
                (
                    ROOT
                    / "results"
                    / "runs"
                    / (
                        "cspa_sgd_seed%d"
                        % seed
                    )
                ),
        }

        for optimizer, run_dir in runs.items():
            row = diagnostics_for_run(
                run_dir
            )

            row.update(
                {
                    "seed":
                        int(seed),

                    "optimizer":
                        optimizer,

                    "run_id":
                        run_dir.name,
                }
            )

            rows.append(
                row
            )

    raw = pd.DataFrame(
        rows
    )

    raw_path = (
        TABLES
        / "cspa_optimizer_robustness.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    numeric = [
        column
        for column
        in raw.columns
        if (
            column
            not in {
                "seed",
                "optimizer",
                "run_id",
            }
            and pd.api.types.is_numeric_dtype(
                raw[
                    column
                ]
            )
        )
    ]

    summary_rows = []

    for optimizer, part in raw.groupby(
        "optimizer",
        sort=False,
    ):
        row = {
            "optimizer":
                optimizer,

            "n":
                int(
                    len(
                        part
                    )
                ),
        }

        for column in numeric:
            row[
                column
                + "_mean"
            ] = float(
                part[
                    column
                ].mean()
            )

            row[
                column
                + "_sd"
            ] = float(
                part[
                    column
                ].std(
                    ddof=1
                )
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        TABLES
        / "cspa_optimizer_robustness_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    f1 = raw.pivot(
        index="seed",
        columns="optimizer",
        values="macro_f1",
    )

    gap = raw.pivot(
        index="seed",
        columns="optimizer",
        values="alignment_gap",
    )

    paired = pd.DataFrame(
        {
            "seed":
                f1.index,

            "delta_macro_f1_sgd_minus_adamw":
                (
                    f1[
                        "SGD_raw"
                    ]
                    - f1[
                        "AdamW"
                    ]
                ).values,

            "delta_alignment_gap_sgd_minus_adamw":
                (
                    gap[
                        "SGD_raw"
                    ]
                    - gap[
                        "AdamW"
                    ]
                ).values,
        }
    )

    paired_path = (
        TABLES
        / "cspa_optimizer_robustness_paired.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    return (
        raw,
        summary,
        paired,
        [
            raw_path,
            summary_path,
            paired_path,
        ],
    )


def plot_optimizer(
    raw: pd.DataFrame,
    extension: str,
):
    order = [
        "AdamW",
        "SGD_raw",
    ]

    summary = (
        raw.groupby(
            "optimizer"
        )
        .agg(
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            macro_f1_sd=(
                "macro_f1",
                "std",
            ),
            alignment_gap_mean=(
                "alignment_gap",
                "mean",
            ),
            alignment_gap_sd=(
                "alignment_gap",
                "std",
            ),
            negative_margin_mean=(
                "negative_directional_margin_rate",
                "mean",
            ),
            negative_margin_sd=(
                "negative_directional_margin_rate",
                "std",
            ),
        )
        .loc[
            order
        ]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            11.0,
            3.8,
        ),
    )

    axes[0].bar(
        order,
        summary[
            "macro_f1_mean"
        ],
        yerr=summary[
            "macro_f1_sd"
        ],
        capsize=4,
    )

    axes[0].set_ylabel(
        "Validation Macro-F1"
    )

    axes[0].set_title(
        "Emotion prediction"
    )

    axes[1].bar(
        order,
        summary[
            "alignment_gap_mean"
        ],
        yerr=summary[
            "alignment_gap_sd"
        ],
        capsize=4,
    )

    axes[1].set_ylabel(
        "Alignment gap"
    )

    axes[1].set_title(
        "Cross-modal alignment"
    )

    axes[2].bar(
        order,
        summary[
            "negative_margin_mean"
        ],
        yerr=summary[
            "negative_margin_sd"
        ],
        capsize=4,
    )

    axes[2].set_ylabel(
        "Negative-margin rate"
    )

    axes[2].set_title(
        "Directional diagnostic"
    )

    fig.suptitle(
        "CSPA optimizer-assumption diagnostic "
        "(3 matched validation seeds)"
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

    output = (
        FIGURES
        / (
            "cspa_optimizer_robustness_validation."
            + extension
        )
    )

    fig.savefig(
        output,
        bbox_inches="tight",
        dpi=220
        if extension
        == "png"
        else None,
    )

    plt.close(
        fig
    )

    return output


def collision_sensitivity(
    audit: pd.DataFrame,
):
    suspicious_verdicts = {
        "exact_duplicate_evidence",
        "partial_duplicate_evidence",
        "high_representation_similarity",
    }

    suspicious = audit[
        audit[
            "verdict"
        ].isin(
            suspicious_verdicts
        )
    ]

    test_ids = set()

    for _, row in suspicious.iterrows():
        if (
            str(
                row[
                    "split_a"
                ]
            )
            == "test"
        ):
            test_ids.add(
                str(
                    row[
                        "song_id_a"
                    ]
                )
            )

        if (
            str(
                row[
                    "split_b"
                ]
            )
            == "test"
        ):
            test_ids.add(
                str(
                    row[
                        "song_id_b"
                    ]
                )
            )

    summary_source = (
        ROOT
        / "results"
        / "summaries"
        / "final_test_runs.csv"
    )

    final = pd.read_csv(
        summary_source
    )

    rows = []

    for _, run in final.iterrows():
        run_id = str(
            run[
                "run_id"
            ]
        )

        prediction_path = (
            ROOT
            / "results"
            / "final_test"
            / run_id
            / "predictions.csv"
        )

        if not prediction_path.exists():
            continue

        predictions = pd.read_csv(
            prediction_path
        )

        filtered = predictions[
            ~predictions[
                "song_id"
            ]
            .astype(str)
            .isin(
                test_ids
            )
        ].copy()

        metrics = classification_metrics(
            filtered[
                "y_true"
            ]
            .astype(int)
            .tolist(),

            filtered[
                "y_pred"
            ]
            .astype(int)
            .tolist(),
        )

        original = float(
            run[
                "macro_f1"
            ]
        )

        revised = float(
            metrics[
                "macro_f1"
            ]
        )

        rows.append(
            {
                "run_id":
                    run_id,

                "experiment":
                    str(
                        run[
                            "experiment"
                        ]
                    ),

                "seed":
                    int(
                        run[
                            "seed"
                        ]
                    ),

                "excluded_test_items":
                    int(
                        len(
                            predictions
                        )
                        - len(
                            filtered
                        )
                    ),

                "original_macro_f1":
                    original,

                "filtered_macro_f1":
                    revised,

                "delta_filtered_minus_original":
                    (
                        revised
                        - original
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    raw_path = (
        TABLES
        / "collision_exclusion_test_sensitivity.csv"
    )

    result.to_csv(
        raw_path,
        index=False,
    )

    if result.empty:
        summary = pd.DataFrame(
            columns=[
                "experiment",
                "n",
                "mean_delta",
                "max_abs_delta",
            ]
        )

    else:
        summary = (
            result.groupby(
                "experiment"
            )
            .agg(
                n=(
                    "seed",
                    "count",
                ),
                mean_delta=(
                    "delta_filtered_minus_original",
                    "mean",
                ),
                max_abs_delta=(
                    "delta_filtered_minus_original",
                    lambda values:
                        float(
                            np.abs(
                                values
                            ).max()
                        ),
                ),
            )
            .reset_index()
        )

    summary_path = (
        TABLES
        / "collision_exclusion_test_sensitivity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return (
        raw_path,
        summary_path,
        sorted(
            test_ids
        ),
    )


def copy_sgd_records():
    copied = []

    for seed in SEEDS:
        source = (
            ROOT
            / "results"
            / "runs"
            / (
                "cspa_sgd_seed%d"
                % seed
            )
        )

        destination = (
            EXTENSION_RUNS
            / (
                "cspa_sgd_seed%d"
                % seed
            )
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name in (
            "config.yaml",
            "system.json",
            "history.csv",
            "gradient_stats.csv",
            "metrics.json",
            "validation_predictions.csv",
            "status.json",
        ):
            src = (
                source
                / name
            )

            if not src.exists():
                continue

            dst = (
                destination
                / name
            )

            shutil.copy2(
                src,
                dst,
            )

            copied.append(
                dst
            )

    return copied


def main() -> int:
    for directory in (
        TABLES,
        FIGURES,
        MANIFESTS,
        EXTENSION_RUNS,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    artifacts = []

    loaded = {}

    projection_rows = []

    for method in METHODS:
        (
            npz_path,
            metadata_path,
            metadata,
            arrays,
        ) = load_representation(
            method
        )

        loaded[
            method
        ] = (
            npz_path,
            metadata_path,
            metadata,
            arrays,
        )

        projection_rows.append(
            {
                "experiment":
                    method,

                "seed":
                    42,

                "split":
                    "validation",

                "n":
                    int(
                        metadata[
                            "n"
                        ]
                    ),

                "checkpoint_sha256":
                    metadata[
                        "checkpoint_sha256"
                    ],

                "preprocessing":
                    "L2 normalization then PCA(max 50) then t-SNE",

                "tsne_perplexity":
                    30,

                "tsne_random_state":
                    42,

                "test_used":
                    False,
            }
        )

        artifacts.extend(
            [
                npz_path,
                metadata_path,
            ]
        )

    projection_manifest = pd.DataFrame(
        projection_rows
    )

    projection_manifest_path = (
        TABLES
        / "representation_projection_manifest.csv"
    )

    projection_manifest.to_csv(
        projection_manifest_path,
        index=False,
    )

    artifacts.append(
        projection_manifest_path
    )

    for extension in (
        "pdf",
        "png",
    ):
        artifacts.append(
            plot_modality_projection(
                loaded,
                extension,
            )
        )

        artifacts.append(
            plot_fused_projection(
                loaded,
                extension,
            )
        )

    (
        optimizer_raw,
        optimizer_summary,
        optimizer_paired,
        optimizer_paths,
    ) = optimizer_diagnostic()

    artifacts.extend(
        optimizer_paths
    )

    for extension in (
        "pdf",
        "png",
    ):
        artifacts.append(
            plot_optimizer(
                optimizer_raw,
                extension,
            )
        )

    collision_path = (
        TABLES
        / "artist_title_collision_audit.csv"
    )

    collision = pd.read_csv(
        collision_path
    )

    artifacts.append(
        collision_path
    )

    (
        sensitivity_path,
        sensitivity_summary_path,
        suspicious_test_ids,
    ) = collision_sensitivity(
        collision
    )

    artifacts.extend(
        [
            sensitivity_path,
            sensitivity_summary_path,
        ]
    )

    verdict_counts = (
        collision[
            "verdict"
        ]
        .value_counts()
        .to_dict()
    )

    resolution = {
        "source_integrity_report":
            "results/manifests/merge_70-15-15_integrity.json",

        "n_flagged_identities":
            int(
                collision[
                    "identity"
                ].nunique()
            ),

        "n_cross_split_pair_comparisons":
            int(
                len(
                    collision
                )
            ),

        "verdict_counts":
            {
                str(key):
                    int(value)
                for key, value
                in verdict_counts.items()
            },

        "suspicious_test_song_ids":
            suspicious_test_ids,

        "interpretation":
            (
                "Artist-title equality alone is treated as a warning. "
                "Duplicate evidence requires raw-audio hash equality, "
                "normalized-lyrics hash equality, or exceptionally high "
                "cached cross-instance representation similarity."
            ),
    }

    resolution_path = (
        MANIFESTS
        / "artist_title_collision_resolution.json"
    )

    resolution_path.write_text(
        json.dumps(
            resolution,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts.append(
        resolution_path
    )

    artifacts.extend(
        copy_sgd_records()
    )

    manifest = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source_git_sha":
            git_sha(),

        "protocol":
            {
                "new_training_runs":
                    3,

                "new_training_seeds":
                    SEEDS,

                "new_training_split":
                    "train/validation only",

                "new_training_test_evaluated":
                    False,

                "sgd_momentum":
                    0.0,

                "sgd_weight_decay":
                    0.0,

                "gradient_clipping":
                    "effectively disabled with threshold 1e9",

                "optimizer_diagnostic_interpretation":
                    (
                        "The SGD experiment tests sensitivity to the "
                        "raw-gradient optimizer assumption. It is not "
                        "a fair optimizer benchmark because the learning "
                        "rate was inherited from the existing tuned "
                        "configuration."
                    ),

                "cspa_theory_scope":
                    (
                        "No certified descent guarantee is claimed. "
                        "The secant beta remains an empirical local "
                        "smoothness estimate rather than a proven "
                        "Lipschitz upper bound."
                    ),

                "representation_split":
                    "validation",

                "representation_seed":
                    42,

                "representation_methods":
                    METHODS,

                "projection":
                    (
                        "L2 normalization -> PCA(max 50 dimensions) "
                        "-> deterministic t-SNE(2 dimensions)"
                    ),

                "tsne_perplexity":
                    30,

                "tsne_random_state":
                    42,

                "collision_posthoc_test_sensitivity":
                    True,

                "collision_test_sensitivity_role":
                    (
                        "diagnostic only; never used for model selection"
                    ),
            },

        "artifacts": [],
    }

    unique = []

    seen = set()

    for artifact in artifacts:
        artifact = Path(
            artifact
        )

        if not artifact.exists():
            raise FileNotFoundError(
                artifact
            )

        resolved = str(
            artifact.resolve()
        )

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        unique.append(
            artifact
        )

        manifest[
            "artifacts"
        ].append(
            {
                "path":
                    str(
                        artifact.relative_to(
                            ROOT
                        )
                    ),

                "size_bytes":
                    int(
                        artifact.stat().st_size
                    ),

                "sha256":
                    sha256_file(
                        artifact
                    ),
            }
        )

    manifest_path = (
        MANIFESTS
        / "small_extension_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("OPTIMIZER-ASSUMPTION DIAGNOSTIC")
    print("=" * 78)

    print(
        optimizer_raw.to_string(
            index=False
        )
    )

    print()
    print("SUMMARY")

    print(
        optimizer_summary.to_string(
            index=False
        )
    )

    print()
    print("PAIRED SGD - ADAMW")

    print(
        optimizer_paired.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("ARTIST/TITLE COLLISION RESOLUTION")
    print("=" * 78)

    print(
        collision.to_string(
            index=False
        )
    )

    print()
    print(
        "Suspicious test IDs:",
        suspicious_test_ids,
    )

    print()
    print(
        "Manifest:",
        manifest_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
