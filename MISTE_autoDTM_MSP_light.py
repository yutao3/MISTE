#!/usr/bin/env python3
"""
MISTE_autoDTM_MSP_light.py: Multi-scale pyramid wrapper for MISTE_autoDTM.py
Implements coarse-to-fine multi-resolution processing by repeatedly calling
MISTE_autoDTM.py on downsampled versions of the input image.

Usage:
  python MISTE_autoDTM_MSP_light.py -i INPUT.tif -r REF.tif -o OUTPUT.tif \
       -m Model_ID -w WEIGHTS.pth [--num_of_scales N] [other MISTE_autoDTM flags]

This script will:
 1) Optionally downsample large input images to multiple scales
 2) For each scale from coarse to full-resolution:
    - Resample input with gdal average
    - Call MISTE_autoDTM.py with the resampled input
      * Use reference DTM only at the coarsest level; thereafter use
        the output DTM from the previous (coarser) level
    - Adjust postprocessing scale factor: (num_of_scales - current_level) * base_scale
    - Store intermediate outputs as <output_base>_level{level}.tif
 3) If input smaller than 1000 px in width or height, runs a single MISTE_autoDTM call

LRD ignores aspect and ref resolution; it simply sets level ratios to (i+1)/N (e.g., for N=3 → 1/3, 2/3, 1.0).

"""
import os
import sys
import argparse
from miste_checkpoints import add_checkpoint_arguments, resolve_checkpoint
import subprocess
import time
from osgeo import gdal

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-scale pyramid wrapper for MISTE_autoDTM.py"
    )
    # Inherit all arguments from MISTE_autoDTM.py
    parser.add_argument('-i', '--input', required=True,
                        help='Input 8-bit 1-channel GeoTIFF')
    parser.add_argument('-r', '--ref', required=True,
                        help='Reference DTM GeoTIFF')
    parser.add_argument('-o', '--output', required=True,
                        help='Output DTM GeoTIFF path')
    add_checkpoint_arguments(parser)
    parser.add_argument('-a', '--asp', default='~/Downloads/ASP/bin',
                        help='Path to ASP bin (dem_mosaic)')
    parser.add_argument('-t', '--tmp', default='data_tmp',
                        help='Temporary working directory')
    # Expert flags
    parser.add_argument('--overlap', type=int, default=280,
                        help='Tile overlap (px)')
    parser.add_argument('--valid_threshold', type=int, default=20,
                        help='Valid pixel intensity threshold')
    parser.add_argument('--max_nodata_pixels', type=int, default=3000,
                        help='Max nodata pixels allowed')
    parser.add_argument('--ndv', type=float,
                        default=-3.40282265508890445e+38,
                        help='NoData value for output tiles')
    parser.add_argument('--scale', type=float, default=2.75,
                        help='Gaussian scale factor for postprocessing')
    parser.add_argument('--inpaint', action='store_true',
                        help='Enable image inpainting on near-valid tiles')
    parser.add_argument('--inpaint_threshold', type=float, default=0.1,
                        help='Min fraction valid pixels to inpaint')
    parser.add_argument('--inpaint_method', choices=['telea','ns'],
                        default='telea', help='Inpainting algorithm')
    parser.add_argument('--fill_smoothing', type=int, default=1,
                        help='Smoothing iterations in FillNodata')
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Inference device (default: cuda)")
    # MSP-specific
    parser.add_argument('--num_of_scales', type=int, default=3,
                        help='Number of scales in multi-scale pyramid')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.weights = resolve_checkpoint(args, parser)
    return args


def call_miste(script_path, inp, ref, out, tmp_dir, args, post_scale):
    """
    Call MISTE_autoDTM.py with forwarded parameters and adjusted scale.
    """
    cmd = [sys.executable, script_path,
           '-i', inp,
           '-r', ref,
           '-o', out,
           '-m', args.model,
           '-w', args.weights,
           '--device', args.device,
           '-a', args.asp,
           '--tmp', tmp_dir,
           '--overlap', str(args.overlap),
           '--valid_threshold', str(args.valid_threshold),
           '--max_nodata_pixels', str(args.max_nodata_pixels),
           '--ndv', str(args.ndv),
           '--scale', str(post_scale),
           '--fill_smoothing', str(args.fill_smoothing)]
    if args.inpaint:
        cmd.append('--inpaint')
        cmd.extend(['--inpaint_threshold', str(args.inpaint_threshold)])
        cmd.extend(['--inpaint_method', args.inpaint_method])
    print(f"[CALL] {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[ERROR] MISTE_autoDTM.py failed at scale with rc={ret.returncode}")
        sys.exit(ret.returncode)


def main():
    start_time = time.time()
    args = parse_args()
    # Absolute paths
    args.asp = os.path.expanduser(args.asp)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    miste_script = os.path.join(script_dir, 'MISTE_autoDTM.py')
    # Ensure tmp parent exists
    os.makedirs(args.tmp, exist_ok=True)
    # Read input size
    ds = gdal.Open(args.input)
    if ds is None:
        print(f"[ERROR] Cannot open input image: {args.input}")
        sys.exit(1)
    width = ds.RasterXSize
    height = ds.RasterYSize
    # Small image: single-scale
    if width < 1000 or height < 1000:
        print("[INFO] Image is too small for multi-scale processing. Running single-scale.")
        # Forward call without MSP
        call_miste(miste_script, args.input, args.ref,
                    args.output, os.path.join(args.tmp, 'single_scale'),
                    args, args.scale)
        print("[INFO] Single-scale processing complete.")
        sys.exit(0)
    # Prepare scales: level 0 (coarse) ... level N-1 (full)
    N = args.num_of_scales
    base_out, ext = os.path.splitext(args.output)
    current_ref = args.ref
    for i in range(N):
        # Compute downsample ratio and postprocess scale
        ds_ratio = float(i+1) / float(N)
        post_scale = (N - i) * args.scale
        level_dir = os.path.join(args.tmp, f'level_{i}')
        os.makedirs(level_dir, exist_ok=True)
        # Downsample input
        lvl_input = os.path.join(level_dir, f'input_level{i}{ext}')
        if ds_ratio < 1.0:
            print(f"[LEVEL {i}] Downsampling input by {ds_ratio:.3f} -> {lvl_input}")
            ds_lvl = gdal.Translate(
                lvl_input, args.input,
                format='GTiff',
                width=int(width * ds_ratio),
                height=int(height * ds_ratio),
                resampleAlg=gdal.GRA_Average
            )
            ds_lvl = None
        else:
            # Full resolution: copy
            print(f"[LEVEL {i}] Copying input (full resolution) -> {lvl_input}")
            ds_lvl = gdal.Translate(
                lvl_input, args.input,
                format='GTiff'
            )
            ds_lvl = None
        # Determine output path
        if i < N-1:
            out_dtm = f"{base_out}_level{i}{ext}"
        else:
            out_dtm = args.output
        print(f"[LEVEL {i}] Calling MISTE_autoDTM with post-scale {post_scale:.3f}")
        call_miste(miste_script, lvl_input, current_ref,
                    out_dtm, level_dir, args, post_scale)
        # Next level reference
        current_ref = out_dtm
    total_elapsed = time.time() - start_time
    print(f"[TOTAL MSP TIME] {total_elapsed:.1f}s across {N} scales")
    print("[ALL DONE] Multi-scale processing complete.")

if __name__ == '__main__':
    main()

