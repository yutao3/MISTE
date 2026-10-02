#!/usr/bin/env python3
"""get_lroc_nac.py  – download selected LROC NAC EDRs from a pre‑built URL list

The script takes:
  1. A **master URL list** (one full https://… link per line).
  2. Either a **file** containing image IDs (one per line) *or* a single
     image ID on the command line.
  3. An **output directory**.

It maps each desired ID to its URL in the master list, then streams the image
into the output directory (skipping those already present).

Examples
--------
Download a batch listed in *wanted.txt*:

    python get_lroc_nac.py ./all_lroc_nac_urls.txt ./wanted.txt ~/lroc_images

Fetch one image directly:

    python get_lroc_nac.py ./all_lroc_nac_urls.txt M1181811415LE ~/lroc_images

Dependencies: only the Python 3 standard library.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Dict

HEADERS = {"User-Agent": "Mozilla/5.0 (get_lroc_nac/1.0)"}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def canonical_id(raw: str) -> str:
    """Return IMG filename in uppercase, ensure suffix."""
    s = raw.strip().upper()
    if not s:
        return ""
    if s.endswith(".IMG"):
        return s
    if s.endswith("LE") or s.endswith("RE"):
        return s + ".IMG"
    return s + "LE.IMG"  # default to left‑eye


def load_url_index(path: Path) -> Dict[str, str]:
    """Read the master list (one URL per line) and map basename → url."""
    mapping: Dict[str, str] = {}
    with path.open() as fp:
        for line in fp:
            url = line.strip()
            if not url:
                continue
            img = Path(url).name.upper()
            mapping[img] = url
    return mapping


def desired_ids(arg: str) -> List[str]:
    p = Path(arg)
    if p.exists():
        ids = [canonical_id(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        ids = [canonical_id(part) for part in arg.split(',')]
    return [i for i in ids if i]


# -----------------------------------------------------------------------------
# Download helper
# -----------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"✓ {dest.name} already exists")
        return
    print(f"↓ {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as src, tmp.open("wb") as out:
            while (chunk := src.read(64 * 1024)):
                out.write(chunk)
        tmp.replace(dest)
        print("✓ Done")
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download selected LROC NAC images from master URL list")
    ap.add_argument("master_list", help="Path to all_lroc_nac_urls.txt")
    ap.add_argument("wanted", help="File with image IDs or a single ID")
    ap.add_argument("output_dir", help="Directory to store images")
    args = ap.parse_args(argv)

    master_path = Path(args.master_list).expanduser().resolve()
    if not master_path.is_file():
        sys.exit(f"Master list not found: {master_path}")

    mapping = load_url_index(master_path)
    ids = desired_ids(args.wanted)
    if not ids:
        sys.exit("No valid image IDs provided")

    out_dir = Path(args.output_dir).expanduser().resolve()

    missing: List[str] = []
    for img_id in ids:
        url = mapping.get(img_id)
        if not url:
            missing.append(img_id)
            print(f"✗ {img_id} not in master list")
            continue
        download(url, out_dir / img_id)

    if missing:
        print("\nImages not found:")
        for m in missing:
            print("  " + m)
        sys.exit(1)

if __name__ == "__main__":
    main()

