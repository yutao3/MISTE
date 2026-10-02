#!/usr/bin/env python3
"""get_latest_lroc_nac_list_and_merge.py  –  v2025‑07‑29

Fetch every available *INDEX/INDEX.TAB* from the LROC NAC archive,
extract the file‐path column, and produce a merged list of download URLs.

Usage
-----
    python get_latest_lroc_nac_list_and_merge.py OUTPUT_DIR [--mirror MIRROR_BASE]

Arguments
~~~~~~~~~
OUTPUT_DIR
    Local directory where INDEX.TAB files will be cached and the merged
    URL list will be written.

--mirror MIRROR_BASE
    Base URL prefix (default:
    "https://pds.lroc.im-ldi.com/data").  The second column in each
    INDEX.TAB entry is concatenated (after stripping spaces/quotes) onto
    this prefix to form the full download URL.

Behavior
--------
* Iterates volumes LROLRC_0001 … LROLRC_0035 (no letter), then LROLRC_0036A/B/C … LROLRC_0999C.
* For each volume, tries HEAD on:

      {MIRROR_BASE}/LRO-L-LROC-2-EDR-V1.0/{volume}/INDEX/INDEX.TAB

  If present (HTTP 200), downloads or refreshes the local copy at:

      OUTPUT_DIR/{volume}/INDEX/INDEX.TAB

  (skips on 404/403).
* After caching all INDEX.TAB files, reads every one, parses each CSV row,
  takes the second field (raw file specification), strips quotes/spaces,
  and writes one full URL per line to:

      OUTPUT_DIR/all_lroc_nac_urls.txt

Dependencies
------------
Only Python 3 standard library.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# -----------------------------------------------------------------------------
# Volume generator
# -----------------------------------------------------------------------------

def iter_volumes() -> Iterator[str]:
    # 0001–0035 no letter
    for i in range(1, 36):
        yield f"LROLRC_{i:04d}"
    # 0036A–0999C
    for i in range(36, 1000):
        for letter in "ABC":
            yield f"LROLRC_{i:04d}{letter}"

# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------

def http_head(url: str) -> Optional[dict[str, str]]:
    try:
        with urlopen(Request(url, method="HEAD")) as r:
            return dict(r.headers)
    except HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise
    except URLError:
        return None

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with urlopen(url) as src, open(tmp, "wb") as out:
            out.write(src.read())
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Download INDEX.TAB files and merge the NAC EDR URLs"
    )
    p.add_argument("output_dir", help="Directory to store INDEX.TAB and merged list")
    p.add_argument(
        "--mirror",
        default="https://pds.lroc.im-ldi.com/data",
        help="Base URL prefix for download links"
    )
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cached_tabs: list[Path] = []
    for vol in iter_volumes():
        remote = f"{args.mirror.rstrip('/')}/LRO-L-LROC-2-EDR-V1.0/{vol}/INDEX/INDEX.TAB"
        hdr = http_head(remote)
        if not hdr:
            continue
        local_path = out_dir / vol / "INDEX" / "INDEX.TAB"
        # Download or refresh unconditionally (no timestamp check)
        print(f"Fetching {vol}/INDEX.TAB")
        try:
            download(remote, local_path)
        except Exception as e:
            print(f"  Failed to fetch {vol}: {e}", file=sys.stderr)
            continue
        cached_tabs.append(local_path)

    if not cached_tabs:
        print("No INDEX.TAB files downloaded. Exiting.")
        sys.exit(1)

    merged_file = out_dir / "all_lroc_nac_urls.txt"
    print(f"Merging {len(cached_tabs)} INDEX.TAB into {merged_file}")
    with merged_file.open("w") as out_fp:
        for tab in sorted(cached_tabs):
            with tab.open(newline="") as fp:
                reader = csv.reader(fp)
                for row in reader:
                    if len(row) < 2:
                        continue
                    # strip quotes and whitespace
                    rel = row[1].strip().strip('"').lstrip('/')
                    url = f"{args.mirror.rstrip('/')}/{rel}"
                    out_fp.write(url + "\n")
    print("Done.")

if __name__ == "__main__":
    main()

