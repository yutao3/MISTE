#!/usr/bin/env python3
"""
MISTE_autoDTM.py: End-to-end geotiff image tiling, DTM inference,
georeferencing, postprocessing, and ASP mosaicing.

Usage:
  python MISTE_autoDTM.py -i INPUT.tif -r REF.tif -o OUTPUT.tif -m Model_ID -w WEIGHTS.pth -a ASP_BIN [options]

DenseDepth (D) is the default inference model. Supply --weights or --body.
Expert flags:
  --overlap             Tile overlap in pixels (default: 280)
  --valid_threshold     Pixel intensities ≥ threshold is considered valid (e.g. 20 for the shadowed region)
  --max_nodata_pixels   Maximum allowed nodata pixels per tile (default: 3000)
  --ndv                 NoData value for georeferenced tiles (default: -3.4e+38)
  --scale               Gaussian decomposition scale factor for postprocessing (default: 3.75)
  --inpaint             Enable nodata inpainting for partial tiles (default: False)
  --inpaint_threshold   Min fraction of valid pixels for inpainting (default: 0.1)
  --inpaint_method      NS or TELEA (opencv)
  --fill_smoothing      Smoothing iterations after reference DTM interpolation
"""

import sys

# Normalize negative scientific-notation NoData arguments before parsing.
new_argv = []
skip = False
for i, tok in enumerate(sys.argv):
    if skip:
        skip = False
        continue
    if tok == '--ndv' and i + 1 < len(sys.argv):
        nxt = sys.argv[i+1]
        # if next token is a negative number, merge
        if nxt.startswith('-') and any(c.isdigit() for c in nxt):
            new_argv.append(f'--ndv={nxt}')
            skip = True
        else:
            new_argv.append(tok)
    else:
        new_argv.append(tok)
sys.argv = new_argv
# End of NoData argument normalization.

import argparse
from miste_checkpoints import add_checkpoint_arguments, resolve_checkpoint
import os
import time
import subprocess
import glob
import shutil
import tempfile
from osgeo import gdal, osr, gdal_array
import numpy as np
import cv2
from natsort import natsorted


def _python_split_listfile(all_txt_path: str, out_prefix: str, lines_per_file: int = 200) -> None:
    """Split a text file into multiple files with a fixed number of lines per file.
    Output files use the prefix `out_prefix` followed by a zero-padded index (e.g., list000, list001)."""
    with open(all_txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        # Create an empty file to keep downstream robust
        open(f"{out_prefix}000", "w").close()
        return
    idx = 0
    for start in range(0, len(lines), lines_per_file):
        chunk = lines[start:start + lines_per_file]
        out_path = f"{out_prefix}{idx:03d}"
        with open(out_path, "w") as fo:
            fo.write("\n".join(chunk) + "\n")
        idx += 1


def validate_geotiff(input_tif, ref_tif):
    """
    Ensure both input and reference files are valid GeoTIFFs with projection metadata.
    Returns True if both are loadable and georeferenced; False otherwise.
    """
    ds_in = gdal.Open(input_tif)
    ds_ref = gdal.Open(ref_tif)
    if ds_in is None or ds_ref is None:
        print("[ERROR] Cannot open input or reference GeoTIFF!")
        return False

    for name, ds in [("input", ds_in), ("reference", ds_ref)]:
        proj = ds.GetProjection()
        gt = ds.GetGeoTransform()
        if not proj or not gt:
            print(f"[ERROR] {name.capitalize()} GeoTIFF missing projection or geotransform.")
            return False
    print("[OK] GeoTIFF validation passed.")
    return True


def make_tiles(input_tif, img_dir, geo_dir,
               overlap, patch_w, patch_h,
               valid_thresh, max_nodata_pixels,
               inpaint, inpaint_thresh, inpaint_method):
    """
    Slice the input image into overlapping patches:
      - Save 3-channel tiles for inference in img_dir
      - Save single-band geoheader tiles in geo_dir
      - Tiles with no nodata are always kept.
      - Tiles with nodata are kept only if:
          a) inpaint=True AND
          b) valid_count >= inpaint_thresh * arr.size
        After inpainting, any tile with nodata_count >= max_nodata_pixels is dropped.
    Returns number of tiles created.
    """
    if not os.path.isfile(input_tif):
        raise FileNotFoundError(f"Input file not found: {input_tif}")
    ds = gdal.Open(input_tif)
    im_h, im_w = ds.RasterYSize, ds.RasterXSize
    if not 0 <= overlap < min(patch_w, patch_h):
        raise ValueError("Overlap must be between 0 and 479 pixels.")
    if im_w < patch_w or im_h < patch_h:
        raise ValueError("Input image must be at least 640 x 480 pixels.")

    # Build offsets including final tiles at right/bottom edges
    step_y = patch_h - overlap
    y_steps = list(range(0, im_h - patch_h + 1, step_y))
    if y_steps[-1] != im_h - patch_h:
        y_steps.append(im_h - patch_h)

    step_x = patch_w - overlap
    x_steps = list(range(0, im_w - patch_w + 1, step_x))
    if x_steps[-1] != im_w - patch_w:
        x_steps.append(im_w - patch_w)

    total_windows = len(y_steps) * len(x_steps)
    print(f"Found {total_windows} potential tiles to inspect.")

    created = 0
    for idx, (y_off, x_off) in enumerate(((y, x)
                                          for y in y_steps
                                          for x in x_steps), 1):
        print(f"\rTiling progress: {idx}/{total_windows}", end="", flush=True)
        mem_ds = gdal.Translate('', ds, format='MEM',
                                srcWin=[x_off, y_off, patch_w, patch_h])
        arr = mem_ds.ReadAsArray()

        valid_count = np.sum(arr >= valid_thresh)
        nodata_count = arr.size - np.count_nonzero(arr)

        # If no nodata, keep immediately
        if nodata_count == 0:
            pass

        else:
            # If tile has nodata, decide if we should inpaint
            if inpaint and valid_count >= inpaint_thresh * arr.size:
                # print(f"\n[Inpaint] Tile {idx}: valid={valid_count}, nodata(before)={nodata_count}")
                mask = (arr == 0).astype('uint8') * 255
                method = cv2.INPAINT_TELEA if inpaint_method == 'telea' else cv2.INPAINT_NS
                arr = cv2.inpaint(arr, mask, 3, method)
                nodata_count = arr.size - np.count_nonzero(arr)
                # print(f"[Inpaint] Tile {idx}: nodata(after) ={nodata_count}")
            else:
                continue  # skip this tile entirely

        # Drop if still too many nodata pixels (in case inpainting fails).
        if nodata_count >= max_nodata_pixels:
            continue

        # Save the tile
        created += 1
        basename = os.path.splitext(os.path.basename(input_tif))[0]
        tile_id = f"{basename}_{created}"

        # 3-band RGB for inference
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        img_out = os.path.join(img_dir, f"{tile_id}.tif")
        cv2.imwrite(img_out, rgb)

        # Geoheader-only copy for georeferencing
        geo_out = os.path.join(geo_dir, f"{tile_id}.tif")
        gdal.GetDriverByName('GTiff').CreateCopy(geo_out, mem_ds)

    print(f"\nTotal tiles created for inference: {created}")
    return created


def get_geo_info(ds):
    """
    Extract geotransform, projection, dimensions, and data type from GDAL dataset.
    Returns (NDV, xsize, ysize, GeoTransform, SpatialRef, DataTypeName).
    """
    band = ds.GetRasterBand(1)
    NDV = band.GetNoDataValue()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    geot = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    dtype = gdal.GetDataTypeName(band.DataType)
    return NDV, xsize, ysize, geot, srs, dtype


def georeference_tile(in_rel, in_header, out_rel, ndv_default):
    """
    Assign georeference from header tile to predicted tile:
      - Reset rotation in geotransform
      - Write output with specified NoData value
    """
    ds_pred = gdal.Open(in_rel)
    arr = ds_pred.ReadAsArray().astype(np.float32)

    ds_hdr = gdal.Open(in_header)
    _, xsize, ysize, geot, srs, _ = get_geo_info(ds_hdr)

    # Zero-rotation geotransform
    geot_new = (geot[0], geot[1], 0.0, geot[3], 0.0, geot[5])

    # Replace NaNs with NDV
    arr[np.isnan(arr)] = ndv_default

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_rel, xsize, ysize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geot_new)
    out_ds.SetProjection(srs.ExportToWkt())
    out_ds.GetRasterBand(1).WriteArray(arr)
    out_ds.GetRasterBand(1).SetNoDataValue(ndv_default)
    out_ds = None
    # print(f"[GeoRef] Wrote {out_rel}")


# --- Tunables (set once, fixed in program) ---
PRED_SCALE_MULT = 1.0   # maps [0,1] -> [-PRED_SCALE_MULT*sf, +PRED_SCALE_MULT*sf]
DETAIL_BOOST    = 12.75   # >1.0 to strengthen high-frequency details in output

def postprocess_tile(ref_tif, in_rel, in_header, out_final, scale_factor, fill_smoothing):
    """
    Merge low-frequency shape from ref_tif with high-frequency detail from in_rel.
    Steps:
      - Resample/Fill ref on the tile footprint
      - Build Gaussian low-pass of ref and pred (σ ≈ scale_factor px)
      - High-pass from pred = pred - lp_pred (zero-mean, variance matched)
      - The input prediction (in_rel) is first linearly scaled from [0,1] to [-1, +1]
        then we do the Gaussian low-/high-pass split and recombine as:
      - Output = lp_ref + DETAIL_BOOST * k * hp_pred
      - Restore nodata from both masks, write georeferenced tile
    Args:
        ref_tif (str): Path to reference DTM (better low frequencies).
        in_rel (str): Path to predicted relative DTM (better high frequencies), assumed in [0,1].
        in_header (str): Path to header/ORI image used for masking (zeros treated as nodata).
        out_final (str): Output path.
        scale_factor (float): Used for Gaussian sigma (px).
        fill_smoothing (int/float): Radius for FillNodata smoothing.

    Returns:
        bool: True on success.
    """
    # --- Open predicted, store GeoT/Proj ---
    ds_pred = gdal.Open(in_rel)
    GeoT = ds_pred.GetGeoTransform()
    proj = ds_pred.GetProjection()
    arr_pred = ds_pred.ReadAsArray().astype(np.float32)

    # --- Resample ref to tile extent ---
    ds_ref = gdal.Open(ref_tif)
    minx, maxy = GeoT[0], GeoT[3]
    resx, resy = GeoT[1], GeoT[5]
    maxx = minx + ds_pred.RasterXSize * resx
    miny = maxy + ds_pred.RasterYSize * resy

    mem_ref = gdal.Translate(
        '', ds_ref, format='MEM',
        resampleAlg=gdal.GRA_CubicSpline,
        xRes=resx, yRes=abs(resy),
        projWin=[minx, maxy, maxx, miny]
    )
    band_ref = mem_ref.GetRasterBand(1)
    NDV_ref = band_ref.GetNoDataValue()
    arr_ref = band_ref.ReadAsArray().astype(np.float32)

    # --- Fill nodata in ref if needed ---
    mask_ref_nodata = (arr_ref == NDV_ref) if NDV_ref is not None else np.zeros_like(arr_ref, dtype=bool)
    if mask_ref_nodata.any():
        gdal.FillNodata(band_ref, None, mem_ref.RasterXSize, int(fill_smoothing))
        arr_ref = band_ref.ReadAsArray().astype(np.float32)

    # --- Header image nodata (zeros in image header) ---
    ds_imghdr = gdal.Open(in_header)
    arr_img = ds_imghdr.ReadAsArray()
    mask_img_nodata = (arr_img == 0)

    # --- Pre-scale prediction from [0,1] to [-PRED_SCALE_MULT*sf, +PRED_SCALE_MULT*sf] ---
    arr_pred = np.clip(arr_pred, 0.0, 1.0)
    # arr_pred_scaled = (arr_pred * 2.0 - 1.0) * (PRED_SCALE_MULT * float(scale_factor))
    arr_pred_scaled = (arr_pred * 2.0 - 1.0) * PRED_SCALE_MULT
    
    # --- Gaussian low-pass helper ---
    def gaussian_lowpass(a: np.ndarray, sigma_px: float) -> np.ndarray:
        k = max(3, int(round(sigma_px * 6)))   # ~6σ covers >99%
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(a, (k, k), sigmaX=sigma_px, sigmaY=sigma_px, borderType=cv2.BORDER_REFLECT101)

    # Cutoff sigma in px from scale_factor (bounded for stability)
    sigma = float(max(1.0, min(64.0, scale_factor)))

    # --- Low-/High-pass decomposition ---
    lp_ref  = gaussian_lowpass(arr_ref,         sigma)          # LF: from true-height ref
    lp_pred = gaussian_lowpass(arr_pred_scaled,  sigma)
    hp_pred = arr_pred_scaled - lp_pred                          # HF: from prediction only

    # --- Valid pixels (exclude any nodata) ---
    valid_mask = (~mask_ref_nodata) & (~mask_img_nodata)
    if np.count_nonzero(valid_mask) < 100:
        out = arr_ref.copy()
    else:
        # Zero-mean the high-pass over valid region to avoid LF leakage
        hp_mean = float(np.mean(hp_pred[valid_mask]))
        hp_pred -= hp_mean

        # Gentle outlier clipping to tame spikes/ringing
        lo, hi = np.percentile(hp_pred[valid_mask], [0.5, 99.5])
        hp_pred = np.clip(hp_pred, lo, hi)

        # Variance matching keeps HF energy plausible relative to ref
        hp_ref  = arr_ref - lp_ref
        std_pred = float(np.std(hp_pred[valid_mask])) + 1e-6
        std_ref  = float(np.std(hp_ref[valid_mask]))  + 1e-6
        k_norm = float(np.clip(std_ref / std_pred, 0.5, 2.0))

        # ---- Final recombination:
        # Low frequency straight from ref (true heights) + boosted high-frequency from prediction.
        out = lp_ref + (DETAIL_BOOST * k_norm) * hp_pred

    # --- Restore nodata and write out ---
    if NDV_ref is None:
        NDV_ref = -3.4028235e38  # GDAL default for Float32 nodata
    out = out.astype(np.float32)
    out[mask_ref_nodata | mask_img_nodata] = NDV_ref

    final_ds = gdal_array.OpenArray(out)
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.CreateCopy(out_final, final_ds, strict=0)
    dst.SetGeoTransform(GeoT)
    dst.SetProjection(proj)
    dst.GetRasterBand(1).SetNoDataValue(NDV_ref)
    dst = None
    return True



def full_chain(input_tif, ref_tif, output_tif, asp_bin, model_id,
               weights_path, tmp_dir,
               overlap, valid_thresh, max_nodata_pixels,
               ndv_default, scale_factor,
               inpaint, inpaint_thresh, inpaint_method,
               fill_smoothing, device="cuda"):
    """
    Execute the full pipeline:
      1) make_tiles  2) inference  3) georeference_tile
      4) postprocess_tile  5) ASP dem_mosaic  6) cleanup
    """
    timers = {}
    
    # Ensure our tmp_dir exists and tell GDAL to use it for all its temporary files
    tmp_dir = os.path.abspath(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="miste_ssp_", dir=tmp_dir)
    gdal.SetConfigOption('CPL_TMPDIR', tmp_dir)

    # 0) Prepare temp folders
    dirs = { 'crop': f"{tmp_dir}/crop", 'geo': f"{tmp_dir}/geoheader",
             'rel': f"{tmp_dir}/relative", 'final': f"{tmp_dir}/final" }
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[Setup] Directory '{name}' => {path}")

    # Validate inputs
    print("[Step 0] Validating input/reference images...")
    if not validate_geotiff(input_tif, ref_tif):
        sys.exit(1)

    # 1) Tiling
    print("[Step 1] Creating tiles...")
    start = time.time()
    n_tiles = make_tiles(
        input_tif, dirs['crop'], dirs['geo'],
        overlap, 640, 480,
        valid_thresh, max_nodata_pixels,
        inpaint, inpaint_thresh, inpaint_method
    )
    if n_tiles == 0:
        print("[ERROR] No tiles created; aborting.")
        sys.exit(1)
    timers['tile_creation'] = time.time() - start
    print(f"[Step 1 Done] {n_tiles} tiles generated in {timers['tile_creation']:.1f}s.")

    # 2) Inference
    print(f"[Step 2] Running inference (Model ID: {model_id})...")
    start = time.time()
    if not os.path.isfile(weights_path):
        print(f"[ERROR] Weights file missing: {weights_path}")
        sys.exit(1)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py"),
        "--model", model_id,
        "--weights", weights_path,
        "--data", f"{dirs['crop']}/",
        "--device", device,
    ]
    if subprocess.run(cmd).returncode != 0:
        print("[ERROR] Inference execution failed.")
        sys.exit(1)
    result_files = glob.glob(f"{dirs['crop']}/*_result.tif")
    if not result_files:
        print("[ERROR] No inference outputs found; check model/weights.")
        sys.exit(1)
    timers['inference'] = time.time() - start
    print(f"[Step 2 Done] {len(result_files)} results in {timers['inference']:.1f}s.")

    # 3) Georeferencing
    print("[Step 3] Georeferencing tiles...")
    start = time.time()
    rel_list = []
    total = len(result_files)
    for idx, res in enumerate(natsorted(result_files), 1):
        base = os.path.splitext(os.path.basename(res))[0][:-7]
        hdr = os.path.join(dirs['geo'], f"{base}.tif")
        out_rel = os.path.join(dirs['rel'], f"{base}_relative.tif")
        print(f"\r[GeoReferencing] {idx}/{total}", end="", flush=True)
        if not os.path.isfile(hdr):
            print(f"\n[ERROR] Missing header: {hdr}")
            sys.exit(1)
        georeference_tile(res, hdr, out_rel, ndv_default)
        rel_list.append(out_rel)
    timers['georeference'] = time.time() - start
    print(f"\n[Step 3 Done] {len(rel_list)} tiles in {timers['georeference']:.1f}s.")

    # 4) Postprocessing
    print("[Step 4] Postprocessing tiles...")
    start = time.time()
    final_tiles = []
    total = len(rel_list)
    for idx, rel in enumerate(rel_list, 1):
        base = os.path.splitext(os.path.basename(rel))[0][:-9]  # strip "_relative"
        # now also point to the matching geoheader tile
        hdr = os.path.join(dirs['geo'], f"{base}.tif")
        out_f = os.path.join(dirs['final'], f"{base}_final.tif")

        print(f"\r[Postprocessing] {idx}/{total}", end="", flush=True)
        if not os.path.isfile(hdr):
            print(f"\n[ERROR] Missing image header for postprocess: {hdr}")
            sys.exit(1)

        if postprocess_tile(ref_tif, rel, hdr, out_f, scale_factor, fill_smoothing) \
           and os.path.isfile(out_f):
            final_tiles.append(out_f)
    if not final_tiles:
        print("\n[ERROR] No postprocessed tiles; aborting.")
        sys.exit(1)
    timers['postprocess'] = time.time() - start
    print(f"\n[Step 4 Done] {len(final_tiles)} tiles in {timers['postprocess']:.1f}s.")

    # 5) Prepare list and split for ASP
    print("[Step 5] Preparing ASP list files...")
    all_txt = os.path.join(tmp_dir, 'all.txt')
    with open(all_txt, 'w') as f:
        for fpath in natsorted(final_tiles):
            f.write(fpath + '\n')
    _python_split_listfile(all_txt, f"{tmp_dir}/list")
    print(f"[Step 5 Done] List files in {tmp_dir}.")

    # 6) ASP mosaicing
    print("[Step 6] Running ASP dem_mosaic...")
    start = time.time()
    dem_exec = os.path.join(asp_bin, 'dem_mosaic')
    if not os.path.isfile(dem_exec):
        print(f"[ERROR] dem_mosaic not found at {dem_exec}")
        sys.exit(1)
    for listfile in glob.glob(f"{tmp_dir}/list*"):
        cmd = [
            dem_exec,
            "-l", listfile,
            "-o", f"{listfile}-mosaic",
            "--erode-length", "12",
            "--threads", "8",
        ]
        if subprocess.run(cmd).returncode != 0:
            print(f"[ERROR] dem_mosaic failed on {listfile}")
            sys.exit(1)
    final_inputs = natsorted(glob.glob(f"{tmp_dir}/list*.tif"))
    if not final_inputs:
        print("[ERROR] No intermediate mosaic tiles found for final dem_mosaic.")
        sys.exit(1)
    cmd = [dem_exec, *final_inputs, "-o", f"{tmp_dir}/final", "--threads", "8"]
    if subprocess.run(cmd).returncode != 0:
        print("[ERROR] Final dem_mosaic step failed.")
        sys.exit(1)
    timers['mosaic'] = time.time() - start
    print(f"[Step 6 Done] Mosaic in {timers['mosaic']:.1f}s.")

    # --- robustly find ASP output & ensure output directory exists ---
    # ASP may produce different final names depending on options/version.
    candidate_paths = [
        os.path.join(tmp_dir, 'final-tile-0.tif'),
        os.path.join(tmp_dir, 'final-DEM.tif'),
        os.path.join(tmp_dir, 'final.tif'),
    ]
    final_tiles = glob.glob(os.path.join(tmp_dir, 'final-tile-*.tif'))
    if len(final_tiles) > 1:
        raise RuntimeError('ASP produced multiple final tiles; combine them into one raster before exporting.')
    src_tile = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if src_tile is None:
        # Fallback: any TIF starting with 'final'
        finals = natsorted(glob.glob(os.path.join(tmp_dir, 'final*.tif')))
        if len(finals) == 1:
            src_tile = finals[0]
        elif len(finals) > 1:
            raise RuntimeError("Multiple final ASP tiles found; combine them before selecting an output.")

    if src_tile is None or not os.path.isfile(src_tile):
        tried = ', '.join(os.path.basename(p) for p in candidate_paths)
        print(f"[ERROR] Expected ASP mosaic output not found in {tmp_dir}. Tried: {tried} and pattern 'final*.tif'.")
        sys.exit(1)

    # Ensure the output directory exists before moving
    out_dir = os.path.dirname(os.path.abspath(output_tif))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Move final output
    shutil.move(src_tile, output_tif)
    

    # Cleanup
    print(f"Cleaning temporary directory {tmp_dir}...")
    shutil.rmtree(tmp_dir)

    # Summary
    print(f"All done! Final DTM: {output_tif}")
    total_time = sum(timers.values())
    print(f"Total elapsed time: {total_time/60:.2f} minutes")
    for step, t in timers.items():
        print(f"  - {step}: {t/60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="MISTE auto DTM: tiling → inference → georef → post → mosaic"
    )
    # Required I/O
    parser.add_argument("-i", "--input", required=True,
                        help="Input 8-bit 1-channel GeoTIFF")
    parser.add_argument("-r", "--ref", required=True,
                        help="Reference DTM GeoTIFF")
    parser.add_argument("-o", "--output", required=True,
                        help="Output DTM GeoTIFF path")
    add_checkpoint_arguments(parser)
    parser.add_argument("-a", "--asp", default="~/Downloads/ASP/bin",
                        help="Path to ASP bin (dem_mosaic)")
    parser.add_argument("-t", "--tmp", default="data_tmp",
                        help="Temporary working directory")
    # Expert params
    parser.add_argument("--overlap", type=int, default=280,
                        help="Tile overlap (px) [expert]")
    parser.add_argument("--valid_threshold", type=int, default=20,
                        help="Requires pixel intensities >= valid_threshold to remove low texture regions under shadow [expert]")
    parser.add_argument("--max_nodata_pixels", type=int, default=3000,
                        help="Max nodata pixels allowed [expert]")
    parser.add_argument("--ndv", type=float, default=-3.40282265508890445e+38,
                        help="NoData value for output tiles [expert]")
    parser.add_argument("--scale", type=float, default=3.75,
                        help="Gaussian scale factor for postprocessing [expert]")
    parser.add_argument("--inpaint", action="store_true",
                        help="[expert] enable image inpainting on near-valid tiles")
    parser.add_argument("--inpaint_threshold", type=float, default=0.1,
                        help="[expert] min fraction valid pixels to inpaint")
    parser.add_argument("--inpaint_method", choices=["telea","ns"], default="telea",
                        help="[expert] inpainting algorithm to use")
    parser.add_argument("--fill_smoothing", type=int, default=1,
                        help="[expert] number of smoothing iterations in FillNodata")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Inference device (default: cuda)")
    args = parser.parse_args()
    args.weights = resolve_checkpoint(args, parser)

    full_chain(
        input_tif=args.input,
        ref_tif=args.ref,
        output_tif=args.output,
        asp_bin=os.path.expanduser(args.asp),
        model_id=args.model,
        weights_path=args.weights,
        tmp_dir=args.tmp,
        overlap=args.overlap,
        valid_thresh=args.valid_threshold,
        max_nodata_pixels=args.max_nodata_pixels,
        ndv_default=args.ndv,
        scale_factor=args.scale,
        inpaint=args.inpaint,
        inpaint_thresh=args.inpaint_threshold,
        inpaint_method=args.inpaint_method,
        fill_smoothing=args.fill_smoothing,
        device=args.device
    )

if __name__ == "__main__":
    main()

