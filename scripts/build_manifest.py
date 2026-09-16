#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from merge_emotion.data.manifest import build_pair_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = ArgumentParser()
    parser.add_argument("--profile", default="70-15-15", choices=["70-15-15", "40-30-30"])
    args = parser.parse_args()
    root = PROJECT_ROOT / "data" / "extracted" / "merge_bimodal_complete"
    output = PROJECT_ROOT / "data" / "processed" / ("pairs_%s.csv" % args.profile)
    frame = build_pair_manifest(root, args.profile, output)
    print("Wrote:", output)
    print(frame.groupby(["split", "quadrant"]).size())
    print("Rows:", len(frame))


if __name__ == "__main__":
    main()
