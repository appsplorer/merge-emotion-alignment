#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


RECORD_ID = "13939205"

DATASETS = {
    "complete": {
        "filename": "MERGE_Bimodal_Complete.zip",
        "md5": "4b4dbe24083a5987e0f37ce4eb1df771",
        "size": 739_234_182,
    },
    "balanced": {
        "filename": "MERGE_Bimodal_Balanced.zip",
        "md5": "df9d1483ee3aacece103f7d5665fd7ec",
        "size": 661_100_000,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="complete",
        help="MERGE bimodal release to download",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Project data directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing archive/extracted directory",
    )
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_size: int) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}

    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if start and response.status_code != 206:
            start = 0
            part.unlink(missing_ok=True)
            response.close()
            return download_file(url, destination, expected_size)

        response.raise_for_status()
        mode = "ab" if start else "wb"

        with part.open(mode) as handle, tqdm(
            total=expected_size,
            initial=start,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))

    part.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()

    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")

        zip_file.extractall(destination)


def main() -> int:
    args = parse_args()
    info = DATASETS[args.dataset]

    root = args.root.resolve()
    raw_dir = root / "raw"
    extract_dir = root / "extracted" / f"merge_bimodal_{args.dataset}"

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.parent.mkdir(parents=True, exist_ok=True)

    archive = raw_dir / info["filename"]
    url = (
        f"https://zenodo.org/records/{RECORD_ID}/files/"
        f"{info['filename']}?download=1"
    )

    if args.force:
        archive.unlink(missing_ok=True)
        archive.with_suffix(archive.suffix + ".part").unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

    if archive.exists():
        current_md5 = md5sum(archive)
        if current_md5 != info["md5"]:
            print("Existing archive failed MD5 verification; removing it.")
            archive.unlink()
        else:
            print(f"Archive already verified: {archive}")

    if not archive.exists():
        print(f"Downloading {info['filename']}")
        print(f"Source: {url}")
        download_file(url, archive, info["size"])

    print("Verifying MD5...")
    actual_md5 = md5sum(archive)
    if actual_md5 != info["md5"]:
        print(
            f"Checksum mismatch: expected {info['md5']}, got {actual_md5}",
            file=sys.stderr,
        )
        return 2

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        print(f"Extracting to {extract_dir}")
        safe_extract(archive, extract_dir)
    else:
        print(f"Extracted directory already exists: {extract_dir}")

    manifest = {
        "dataset": f"MERGE_Bimodal_{args.dataset.title()}",
        "zenodo_record": RECORD_ID,
        "filename": info["filename"],
        "url": url,
        "md5": actual_md5,
        "archive_bytes": archive.stat().st_size,
        "archive_path": str(archive),
        "extract_path": str(extract_dir),
        "verified_at_unix": int(time.time()),
    }

    manifest_path = raw_dir / f"merge_bimodal_{args.dataset}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    files = sum(1 for path in extract_dir.rglob("*") if path.is_file())
    print(f"Verified: {actual_md5}")
    print(f"Extracted files: {files}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
