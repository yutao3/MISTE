#!/usr/bin/env python3
"""
pre_adjust_brightness.py – Shadow‑aware tone mapping for 8‑bit GeoTIFFs
=======================================================================

Compatible with **Python ≥ 3.7**  |  Dependencies: `rasterio`, `numpy`, `opencv‑python`

Changelog
---------
* **v0.4 (2025‑05‑05)**  
  • *mask_gamma*: now uses a **ramp mask** (two quantiles) so mildly dark areas
    are brightened too; param `--shadow-high-quantile`.  
  • *sigmoid*: ignores NoData when rescaling → no more "all dark" results on
    sparse scenes.  
  • Added clear terminal progress messages.
* **v0.3 (2025‑05‑05)** – `mask_gamma` fixed to ignore NoData in thresholding.
* **v0.2** – Added `mask_gamma` local method.  
* **v0.1** – Initial release with `clahe`, `agc`, `sigmoid`.

Usage
-----
```bash
# Local shadow brightening with ramp
python pre_adjust_brightness.py in.tif out.tif -m mask_gamma \
       --shadow-quantile 0.25 --shadow-high-quantile 0.45 --gamma 0.45 --sigma 20

# Updated sigmoid (global S‑curve)
python pre_adjust_brightness.py in.tif out_sigmoid.tif -m sigmoid --gain 8 --cutoff 0.45
```
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import numpy as np
import rasterio

###############################################################################
# Helper functions
###############################################################################


def _progress(msg: str):
    """Lightweight progress reporter."""
    print(f"[pre_adjust_brightness] {msg}")


def _read_src(path: str) -> Tuple[np.ndarray, dict, Optional[int]]:
    """Read first band of GeoTIFF, return uint8 image, profile, nodata."""
    _progress("Reading input file …")
    with rasterio.open(path) as src:
        img = src.read(1, masked=False)
        profile = src.profile
        nodata = src.nodata

    # Scale to 8‑bit if necessary
    if img.dtype != np.uint8:
        _progress("Converting to 8‑bit …")
        if nodata is not None:
            valid = img[img != nodata]
        else:
            valid = img
        if valid.size == 0:
            raise RuntimeError("Image contains only NoData pixels.")
        v_min, v_max = float(valid.min()), float(valid.max())
        if v_max == v_min:
            v_max += 1  # avoid /0 on constant images
        img = np.clip((img - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)
        profile.update(dtype="uint8")
        if nodata is not None and not (0 <= nodata <= 255):
            nodata = None

    return img, profile, nodata


def _write_tif(path: str, img: np.ndarray, profile: dict, nodata):
    _progress("Writing output file …")
    profile = profile.copy()
    profile.update(dtype="uint8", count=1, compress="deflate")
    if nodata is not None:
        profile.update(nodata=int(nodata))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(img, 1)
    _progress("Done.")

###############################################################################
# Global transforms
###############################################################################


def clahe_adjust(img: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    import cv2
    _progress("Applying CLAHE …")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def agc_adjust(img: np.ndarray) -> np.ndarray:
    _progress("Applying adaptive gamma …")
    mean = np.clip(img.mean() / 255.0, 1e-3, 0.999)
    gamma = np.log(0.5) / np.log(mean)
    return np.uint8(((img / 255.0) ** gamma) * 255)


def sigmoid_adjust(img: np.ndarray, gain: float, cutoff: float, nodata: Optional[int]) -> np.ndarray:
    _progress("Applying sigmoid S‑curve …")
    x = img.astype(np.float32) / 255.0
    y = 1.0 / (1.0 + np.exp(gain * (cutoff - x)))

    # Rescale y based on **valid** pixels only
    if nodata is not None:
        valid = y[img != nodata]
    else:
        valid = y
    y_min, y_max = float(valid.min()), float(valid.max())
    if y_max == y_min:
        y_max += 1e-6
    y = (y - y_min) / (y_max - y_min)
    out = np.uint8(y * 255)
    if nodata is not None:
        out[img == nodata] = nodata
    return out

###############################################################################
# Local (shadow‑aware) transform
###############################################################################


def mask_gamma_adjust(
    img: np.ndarray,
    *,
    nodata: Optional[int],
    shadow_quantile: float,
    shadow_high_quantile: float,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    """Brighten shadows using a **ramp mask** between two quantiles.

    • Pixels below `shadow_quantile` → weight 1 (fully brightened).  
    • Pixels above `shadow_high_quantile` → weight 0 (unchanged).  
    • Linear blend in‑between → mildly dark regions get partial lift.
    """
    import cv2

    _progress("Computing shadow mask …")
    img_f = img.astype(np.float32) / 255.0

    # Exclude nodata from statistics
    if nodata is not None:
        valid = img_f[img != nodata]
    else:
        valid = img_f
    if valid.size == 0:
        return img

    low_thr = np.quantile(valid, shadow_quantile)
    high_thr = np.quantile(valid, shadow_high_quantile)
    if high_thr <= low_thr:
        high_thr = low_thr + 1e-3  # ensure proper ramp

    # Piece‑wise linear mask 0‑1
    mask = np.zeros_like(img_f, dtype=np.float32)
    # fully shadow
    mask[img_f <= low_thr] = 1.0
    # ramp
    ramp_idx = (img_f > low_thr) & (img_f < high_thr)
    mask[ramp_idx] = (high_thr - img_f[ramp_idx]) / (high_thr - low_thr)

    if nodata is not None:
        mask[img == nodata] = 0.0

    # Feather
    mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    mask = np.clip(mask, 0.0, 1.0)

    _progress("Applying gamma inside shadows …")
    bright = img_f ** gamma  # γ < 1 brightens
    out = mask * bright + (1.0 - mask) * img_f
    out = np.uint8(np.clip(out * 255.0, 0, 255))
    if nodata is not None:
        out[img == nodata] = nodata
    return out

###############################################################################
# CLI
###############################################################################

def main():
    ap = argparse.ArgumentParser(description="Shadow‑aware brightness stretching for greyscale GeoTIFFs.")
    ap.add_argument("input", help="Input GeoTIFF")
    ap.add_argument("output", help="Output GeoTIFF")
    ap.add_argument("-m", "--method", choices=["clahe", "agc", "sigmoid", "mask_gamma"], required=True)

    # clahe
    ap.add_argument("--clip-limit", type=float, default=4.0)
    ap.add_argument("--tile-size", type=int, default=64)

    # sigmoid
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--cutoff", type=float, default=0.5)

    # mask_gamma
    ap.add_argument("--shadow-quantile", type=float, default=0.25, help="Lower quantile for full lift (0‑1)")
    ap.add_argument("--shadow-high-quantile", type=float, default=0.45, help="Upper quantile where lift stops (0‑1)")
    ap.add_argument("--gamma", type=float, default=0.5, help="Gamma (<1 brightens) applied to shadows")
    ap.add_argument("--sigma", type=float, default=15.0, help="Gaussian blur σ for mask feathering (px)")

    args = ap.parse_args()

    img, profile, nodata = _read_src(args.input)

    if args.method == "clahe":
        img_out = clahe_adjust(img, args.clip_limit, args.tile_size)
    elif args.method == "agc":
        img_out = agc_adjust(img)
    elif args.method == "sigmoid":
        img_out = sigmoid_adjust(img, args.gain, args.cutoff, nodata)
    elif args.method == "mask_gamma":
        img_out = mask_gamma_adjust(
            img,
            nodata=nodata,
            shadow_quantile=args.shadow_quantile,
            shadow_high_quantile=args.shadow_high_quantile,
            gamma=args.gamma,
            sigma=args.sigma,
        )
    else:
        sys.exit("Unknown method")

    _write_tif(args.output, img_out, profile, nodata)


if __name__ == "__main__":
    main()

