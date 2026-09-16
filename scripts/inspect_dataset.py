#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from merge_emotion.data.integrity import (
    file_inventory,
    inspect_table,
    tabular_files,
    validate_merge_dataset,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "extracted"
    / "merge_bimodal_complete"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect MERGE and validate "
            "split integrity."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Extracted MERGE dataset "
            "directory."
        ),
    )

    parser.add_argument(
        "--profile",
        default="70-15-15",
        choices=(
            "70-15-15",
            "40-30-30",
        ),
        help=(
            "Official TVT profile "
            "to validate."
        ),
    )

    return parser.parse_args()


def save_split_plot(
    counts,
    png_path,
    pdf_path,
):
    order = [
        "train",
        "validation",
        "test",
    ]

    pivot = (
        counts
        .pivot(
            index="quadrant",
            columns="split",
            values="count",
        )
        .fillna(0)
    )

    available_order = [
        split
        for split in order
        if split in pivot.columns
    ]

    pivot = pivot[
        available_order
    ]

    ax = pivot.plot(
        kind="bar",
        figsize=(
            8.0,
            5.0,
        ),
    )

    ax.set_title(
        "MERGE Bimodal Complete: "
        "class distribution by split"
    )

    ax.set_xlabel(
        "Emotion quadrant"
    )

    ax.set_ylabel(
        "Number of songs"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    figure = ax.get_figure()
    figure.tight_layout()

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main():
    args = parse_args()

    dataset_root = (
        args.root.resolve()
    )

    if not dataset_root.exists():
        print(
            "Dataset directory "
            f"does not exist: "
            f"{dataset_root}",
            file=sys.stderr,
        )

        return 2

    manifest_dir = (
        PROJECT_ROOT
        / "results"
        / "manifests"
    )

    summary_dir = (
        PROJECT_ROOT
        / "results"
        / "summaries"
    )

    figure_dir = (
        PROJECT_ROOT
        / "figures"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "MERGE DATASET INSPECTION"
    )

    print(
        "=" * 72
    )

    print(
        f"Root: {dataset_root}"
    )

    print(
        f"TVT profile: "
        f"{args.profile}"
    )

    print()

    inventory = file_inventory(
        dataset_root
    )

    inventory_path = (
        manifest_dir
        / (
            "merge_bimodal_complete_"
            "inventory.json"
        )
    )

    inventory_path.write_text(
        json.dumps(
            inventory,
            indent=2,
        )
        + "\n"
    )

    print(
        f"Files: "
        f"{inventory['files']:,}"
    )

    print(
        "Size: "
        f"{inventory['bytes'] / (1024 ** 3):.3f} GiB"
    )

    print()
    print("File types:")

    for (
        extension,
        stats,
    ) in (
        inventory[
            "extensions"
        ].items()
    ):
        print(
            f"  {extension:12s} "
            f"{stats['files']:6d} "
            "files "
            f"{stats['bytes'] / (1024 ** 2):9.2f} "
            "MiB"
        )

    tables = [
        inspect_table(path)
        for path
        in tabular_files(
            dataset_root
        )
    ]

    table_manifest_path = (
        manifest_dir
        / (
            "merge_bimodal_complete_"
            "tables.json"
        )
    )

    table_manifest_path.write_text(
        json.dumps(
            tables,
            indent=2,
        )
        + "\n"
    )

    print()
    print("Tabular files:")

    for table in tables:
        path = Path(
            table["path"]
        )

        relative = path.relative_to(
            dataset_root
        )

        if table["readable"]:
            print(
                f"  {relative} "
                f"[{table['rows']} rows]"
            )

            print(
                "    columns: "
                + ", ".join(
                    table["columns"]
                )
            )

        else:
            print(
                f"  {relative} "
                "[READ ERROR: "
                f"{table['error']}]"
            )

    print()
    print(
        "=" * 72
    )

    print(
        "SPLIT INTEGRITY / "
        "LEAKAGE AUDIT"
    )

    print(
        "=" * 72
    )

    try:
        report, counts = (
            validate_merge_dataset(
                dataset_root,
                profile=args.profile,
            )
        )

    except Exception as exc:
        print(
            "Validation could not "
            f"complete: {exc}",
            file=sys.stderr,
        )

        print()

        print(
            "Inventory and table "
            "manifests were saved."
        )

        return 3

    report_path = (
        manifest_dir
        / (
            f"merge_{args.profile}"
            "_integrity.json"
        )
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    counts_path = (
        summary_dir
        / (
            f"merge_{args.profile}"
            "_class_counts.csv"
        )
    )

    counts.to_csv(
        counts_path,
        index=False,
    )

    plot_stem = (
        figure_dir
        / (
            "dataset_distribution_"
            f"{args.profile}"
        )
    )

    save_split_plot(
        counts,
        plot_stem.with_suffix(
            ".png"
        ),
        plot_stem.with_suffix(
            ".pdf"
        ),
    )

    print(
        f"Metadata: "
        f"{report['metadata_file']}"
    )

    print(
        "Metadata split-ID column: "
        f"{report['metadata_split_id_column']}"
    )

    print(
        "Metadata audio-ID column: "
        f"{report['metadata_audio_id_column']}"
    )

    print(
        "Metadata lyric-ID column: "
        f"{report['metadata_lyric_id_column']}"
    )

    print(
        f"Rows: "
        f"{report['metadata_rows']}"
    )

    print(
        "Unique song IDs: "
        f"{report['metadata_unique_ids']}"
    )

    print()
    print("Quadrant counts:")

    for (
        quadrant,
        count,
    ) in (
        report[
            "quadrant_counts"
        ].items()
    ):
        print(
            f"  {quadrant}: "
            f"{count}"
        )

    print()
    print("Split files:")

    for (
        split,
        path,
    ) in (
        report[
            "split_files"
        ].items()
    ):
        print(
            f"  {split:10s}: "
            f"{path}"
        )

    print()
    print("Split sizes:")

    for (
        split,
        size,
    ) in (
        report[
            "split_sizes"
        ].items()
    ):
        ratio = (
            100
            * report[
                "split_ratios"
            ][split]
        )

        print(
            f"  {split:10s}: "
            f"{size:4d} "
            f"({ratio:6.2f}%)"
        )

    print()
    print(
        "Direct ID overlaps:"
    )

    for (
        pair,
        values,
    ) in (
        report[
            "direct_id_overlaps"
        ].items()
    ):
        print(
            f"  {pair}: "
            f"{len(values)}"
        )

    print()
    print(
        "Canonical identity "
        "overlaps:"
    )

    for (
        pair,
        values,
    ) in (
        report[
            "canonical_id_overlaps"
        ].items()
    ):
        print(
            f"  {pair}: "
            f"{len(values)}"
        )

    print()

    print(
        "Artist/title cross-split "
        "candidates: "
        f"{len(report['artist_title_cross_split_collisions'])}"
    )

    print(
        "Audio files discovered: "
        f"{report['audio_files_found']}"
    )

    print(
        "Lyrics files discovered: "
        f"{report['lyrics_files_found']}"
    )

    print(
        "Expected audio IDs: "
        f"{report['expected_audio_ids']}"
    )

    print(
        "Expected lyric IDs: "
        f"{report['expected_lyric_ids']}"
    )

    print(
        "Audio IDs matched by stem: "
        f"{report['audio_ids_matched_by_stem']}"
    )

    print(
        "Lyric IDs matched by stem: "
        f"{report['lyrics_ids_matched_by_stem']}"
    )

    if report["warnings"]:
        print()
        print("Warnings:")

        for warning in (
            report["warnings"]
        ):
            print(
                f"  - {warning}"
            )

    if report["hard_errors"]:
        print()
        print("FAILED checks:")

        for error in (
            report[
                "hard_errors"
            ]
        ):
            print(
                f"  - {error}"
            )

    print()

    print(
        f"Inventory: "
        f"{inventory_path}"
    )

    print(
        f"Tables:    "
        f"{table_manifest_path}"
    )

    print(
        f"Audit:     "
        f"{report_path}"
    )

    print(
        f"Counts:    "
        f"{counts_path}"
    )

    print(
        "Plot:      "
        f"{plot_stem.with_suffix('.png')}"
    )

    print()

    print(
        "FINAL STATUS: "
        f"{report['status']}"
    )

    return (
        0
        if report["status"]
        == "PASS"
        else 4
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )