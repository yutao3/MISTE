#!/usr/bin/env python3
"""
Script to identify low contrast GeoTIFF images in a directory.
Usage:
    python check_low_quality_robust.py <input_directory> <output_text_file>

This script flags an image as low contrast if multiple indicators are below threshold, including:
Local standard deviation (on patches).
Histogram spread (e.g., 95th - 5th percentile).
Image entropy.
Laplacian energy.

"""
import os
import sys
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from skimage.filters import laplace
from skimage.util import view_as_windows, img_as_float
from skimage.exposure import histogram
from scipy.stats import entropy as scipy_entropy

PATCH_SIZE = 64
MAX_DIM = 512

def compute_entropy(image):
    hist, _ = histogram(image)
    hist = hist / np.sum(hist)
    return scipy_entropy(hist, base=2)

def compute_histogram_spread(image):
    p5 = np.percentile(image, 5)
    p95 = np.percentile(image, 95)
    return p95 - p5

def compute_local_std(image, patch_size):
    h, w = image.shape
    if h < patch_size or w < patch_size:
        return np.std(image)
    try:
        patches = view_as_windows(image, (patch_size, patch_size), step=patch_size)
        stds = [np.std(patch) for row in patches for patch in row]
        return np.mean(stds)
    except Exception:
        return np.std(image)

def compute_laplacian_mean(image):
    lap = laplace(image)
    return np.mean(np.abs(lap))

def compute_image_metrics(image):
    image = img_as_float(image)
    std_val = compute_local_std(image, PATCH_SIZE)
    entropy_val = compute_entropy(image)
    hist_spread_val = compute_histogram_spread(image)
    hist_spread_norm = hist_spread_val / (np.max(image) - np.min(image) + 1e-6)
    lap_mean = compute_laplacian_mean(image)
    return std_val, entropy_val, hist_spread_norm, lap_mean

def process_image(file_path):
    try:
        with rasterio.open(file_path) as src:
            width, height = src.width, src.height
            scale = max(width, height) / MAX_DIM
            if scale < 1:
                scale = 1
            out_shape = (int(height / scale), int(width / scale))
            img = src.read(1, out_shape=out_shape, resampling=Resampling.average)
            metrics = compute_image_metrics(img)
            return metrics
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def classify_quality_adaptive(metrics, avg, stddev):
    """
    Adaptive classification:
    - If 2+ metrics are > 1 std below avg → low
    - If all within ±1 std → mid
    - If 2+ metrics > 1 std above avg → high
    """
    below = sum([(m < a - s) for m, a, s in zip(metrics, avg, stddev)])
    above = sum([(m > a + s) for m, a, s in zip(metrics, avg, stddev)])

    if below >= 2:
        return "low"
    elif above >= 2:
        return "high"
    else:
        return "mid"

def main():
    if len(sys.argv) != 3:
        print("Usage: python check_low_quality.py <input_directory> <output_text_file>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    tif_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    if not tif_files:
        print("No GeoTIFF files found.")
        sys.exit(1)

    print(f"Found {len(tif_files)} image(s). Processing...")

    results = []
    for file_path in tif_files:
        print(f"Processing: {file_path}")
        metrics = process_image(file_path)
        if metrics:
            results.append((os.path.abspath(file_path), *metrics))

    if not results:
        print("No valid images processed.")
        return

    # Extract metrics
    metrics_array = np.array([r[1:5] for r in results])
    avg_metrics = np.mean(metrics_array, axis=0)
    std_metrics = np.std(metrics_array, axis=0)

    # Classify each image
    results_with_quality = []
    for row in results:
        filename, metrics = row[0], row[1:5]
        quality = classify_quality_adaptive(metrics, avg_metrics, std_metrics)
        results_with_quality.append((filename, *metrics, quality))

    # Write output file
    with open(output_file, 'w') as f:
        f.write("Filename\tLocal_STD\tEntropy\tHistSpread\tLaplacianMean\tQuality\n")
        for row in results_with_quality:
            f.write(f"{row[0]}\t{row[1]:.4f}\t{row[2]:.4f}\t{row[3]:.4f}\t{row[4]:.6f}\t{row[5]}\n")
        f.write("\nAVERAGE\t")
        f.write(f"{avg_metrics[0]:.4f}\t{avg_metrics[1]:.4f}\t{avg_metrics[2]:.4f}\t{avg_metrics[3]:.6f}\t-\n")

    print(f"\nDone. Results written to: {output_file}")

if __name__ == "__main__":
    main()
