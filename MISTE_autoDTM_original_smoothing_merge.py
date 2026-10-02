#!/usr/bin/env python3
"""
MISTE_autoDTM_original_smoothing_merge.py: End-to-end geotiff image tiling, DTM inference,
georeferencing, postprocessing, and ASP mosaicing.

Usage:
  python MISTE_autoDTM_original_smoothing_merge.py -i INPUT.tif -r REF.tif -o OUTPUT.tif -m Model_ID -w WEIGHTS.pth -a ASP_BIN [options]

DenseDepth (D) is the default inference model. Supply --weights or --body.
Expert flags:
  --overlap             Tile overlap in pixels (default: 280)
  --valid_threshold     Pixel intensities ≥ threshold is considered valid (e.g. 20 for the shadowed region)
  --max_nodata_pixels   Maximum allowed nodata pixels per tile (default: 3000)
  --ndv                 NoData value for georeferenced tiles (default: -3.4e+38)
  --scale               ± scale factor for postprocessing (default: 1.25)
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
from osgeo import gdal, osr, gdal_array
import numpy as np
import cv2
from natsort import natsorted


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


def postprocess_tile(ref_tif, in_rel, in_header, out_final, scale_factor, fill_smoothing):
    """
    Compare predicted tile to reference and input image:
      - Resample reference to tile extent
      - Fill nodata in reference via interpolation
      - Apply scale, compute difference, smooth artifacts
      - Restore original nodata mask from BOTH:
          * the reference DTM
          * the input image geoheader tile
      - Save final, georeferenced DTM tile
    Returns True if successful.
    """
    # Open predicted DTM
    ds_pred = gdal.Open(in_rel)
    GeoT = ds_pred.GetGeoTransform()
    proj = ds_pred.GetProjection()

    # 1) Resample reference tile
    ds_ref = gdal.Open(ref_tif)
    minx, maxy = GeoT[0], GeoT[3]
    res = GeoT[1]
    maxx = minx + ds_pred.RasterXSize * res
    miny = maxy + ds_pred.RasterYSize * GeoT[5]
    mem_ref = gdal.Translate('', ds_ref, format='MEM',
                             resampleAlg=gdal.GRA_CubicSpline,
                             xRes=res, yRes=res,
                             projWin=[minx, maxy, maxx, miny])
    band_ref = mem_ref.GetRasterBand(1)
    NDV_ref = band_ref.GetNoDataValue()
    arr_ref = band_ref.ReadAsArray()

    # Mask where ref is nodata
    mask_ref_nodata = (arr_ref == NDV_ref)
    if mask_ref_nodata.any():
        # fill those holes up to full tile radius
        max_search = mem_ref.RasterXSize
        gdal.FillNodata(band_ref, None, max_search, fill_smoothing)
        arr_ref = band_ref.ReadAsArray()

    # 2) Read input‐image geoheader tile to find its nodata
    ds_imghdr = gdal.Open(in_header)
    arr_img = ds_imghdr.ReadAsArray()
    # assume zeros in the image tile are nodata
    mask_img_nodata = (arr_img == 0)

    # 3) Scale prediction
    mem_pred = gdal.Translate('', ds_pred, format='MEM',
                              scaleParams=[[0, 1, -scale_factor, scale_factor]])
    arr_pred = mem_pred.GetRasterBand(1).ReadAsArray()

    # 4) Compute diff & smooth
    diff_arr = arr_pred - arr_ref
    diff_ds = gdal_array.OpenArray(diff_arr)
    down = gdal.Translate('', diff_ds, format='MEM',
                           width=round(diff_ds.RasterXSize * 0.05),
                           height=round(diff_ds.RasterYSize * 0.05),
                           resampleAlg=gdal.GRA_Average)
    up = gdal.Translate('', down, format='MEM',
                         width=diff_ds.RasterXSize,
                         height=diff_ds.RasterYSize,
                         resampleAlg=gdal.GRA_CubicSpline)

    final_arr = arr_pred - up.GetRasterBand(1).ReadAsArray()

    # 5) Restore nodata from both masks
    final_arr[mask_ref_nodata] = NDV_ref
    final_arr[mask_img_nodata] = NDV_ref

    # 6) Write out
    final_ds = gdal_array.OpenArray(final_arr)
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.CreateCopy(out_final, final_ds)
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
               fill_smoothing):
    """
    Execute the full pipeline:
      1) make_tiles  2) inference  3) georeference_tile
      4) postprocess_tile  5) ASP dem_mosaic  6) cleanup
    """
    timers = {}
    
    # Ensure our tmp_dir exists and tell GDAL to use it for all its temporary files
    tmp_dir = os.path.abspath(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
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
    cmd = f"python inference.py --model {model_id} --weights {weights_path} --data {dirs['crop']}/"
    if subprocess.run(cmd, shell=True).returncode != 0:
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
        for fpath in natsorted(final_tiles): f.write(fpath + '\n')
    subprocess.run(f"split -l 200 {all_txt} {tmp_dir}/list", shell=True)
    print(f"[Step 5 Done] List files in {tmp_dir}.")

    # 6) ASP mosaicing
    print("[Step 6] Running ASP dem_mosaic...")
    start = time.time()
    dem_exec = os.path.join(asp_bin, 'dem_mosaic')
    if not os.path.isfile(dem_exec):
        print(f"[ERROR] dem_mosaic not found at {dem_exec}")
        sys.exit(1)
    for listfile in glob.glob(f"{tmp_dir}/list*"):
        cmd = f"{dem_exec} -l {listfile} -o {listfile}-mosaic --erode-length 12 --threads 8"
        if subprocess.run(cmd, shell=True).returncode != 0:
            print(f"[ERROR] dem_mosaic failed on {listfile}")
            sys.exit(1)
    cmd = f"{dem_exec} {tmp_dir}/list*.tif -o {tmp_dir}/final --threads 8"
    if subprocess.run(cmd, shell=True).returncode != 0:
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
    src_tile = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if src_tile is None:
        # Fallback: any TIF starting with 'final'
        finals = natsorted(glob.glob(os.path.join(tmp_dir, 'final*.tif')))
        if len(finals) == 1:
            src_tile = finals[0]
        elif len(finals) > 1:
            # pick the newest by mtime
            src_tile = max(finals, key=lambda p: os.path.getmtime(p))

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
    parser.add_argument("--scale", type=float, default=12.75,
                        help="± scale factor for postprocessing [expert]")
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
        fill_smoothing=args.fill_smoothing
    )

if __name__ == "__main__":
    main()

