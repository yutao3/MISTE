#!/usr/bin/env python3
"""
Script to combine a reference DEM (absolute) with a high-frequency relative DEM,
using a polynomial surface fit to align large-scale slope/offset.

Features:
  - Checks for NoData in the reference tile, optionally skips processing if present.
  - Optional normalization of the relative DEM from [0..1] to [-1..+1].
  - A scale factor applied BEFORE computing difference with the reference.
  - Automatic polynomial order selection (1 vs. 2) based on residual error.
  - Rescales final output to [ref_min - 1.5, ref_max + 1.5] in an example offset.

Usage:
    python postprocess_poly_auto_scale.py <input_ref> <input_relative> <output_final> \
                                    [<scale_factor>] [<do_normalize>]

Examples:
    1) No scaling, no normalization, auto order selection:
       python postprocess_poly_auto_scale.py ref_dem.tif rel_dem.tif fused_dem.tif

    2) Scale=5.0, no normalization, auto order selection:
       python postprocess_poly_auto_scale.py ref_dem.tif rel_dem.tif fused_dem.tif 5.0

    3) Scale=10, with normalization, auto order selection:
       python postprocess_poly_auto_scale.py ref_dem.tif rel_dem.tif fused_dem.tif 10 True
"""

import sys
import numpy as np
from osgeo import gdal

def fit_polynomial_surface(x, y, z, order=1):
    """
    Fit a 2D polynomial of given 'order' to data z = f(x, y).
    Returns coefficients in a flattened form.

    For order=1: z = c0 + c1*x + c2*y  (plane).
    For order=2: z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2.
    """
    if order not in [1, 2]:
        raise ValueError("Only order=1 or order=2 are supported in this example.")

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    if order == 1:
        # Plane: a + b*x + c*y
        M = np.column_stack([np.ones_like(x), x, y])
    else:
        # Quadratic: a + b*x + c*y + d*x^2 + e*x*y + f*y^2
        M = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])

    coeffs, residuals, rank, s = np.linalg.lstsq(M, z, rcond=None)
    return coeffs

def evaluate_polynomial_surface(x, y, coeffs, order=1):
    """
    Evaluate the 2D polynomial with given coefficients at coordinates (x, y).
    """
    if order == 1:
        a, b, c = coeffs  # plane
        return a + b*x + c*y
    else:
        a, b, c, d, e, f = coeffs  # quadratic
        return a + b*x + c*y + d*x**2 + e*x*y + f*y**2

def compute_rmse(a, b):
    """
    Compute the root mean square error between arrays a and b.
    """
    diff = a - b
    mse = np.mean(diff**2)
    return np.sqrt(mse)

def postprocess(input_ref, input_relative, output_final, 
                scale_factor=1.0, do_normalize=False):
    """
    Main workflow:
      1. Align/clip reference to match relative DEM's extent/resolution.
      2. Check NoData in the clipped reference tile; skip if needed.
      3. (Optional) Normalize relative DEM from [0..1] to [-1..+1].
      4. Scale the relative DEM by 'scale_factor'.
      5. difference = reference - scaled_relative
      6. Fit polynomial (order=1 and order=2), pick best by RMSE.
      7. final = scaled_relative + difference_poly
      8. Rescale final to [ref_min - 1.5, ref_max + 1.5] (example).
      9. Save final with the same georeferencing, using the reference's NDV if available.
    """
    # ---------------------------------------------------------------------
    # 1) Open relative DEM and get bounding box
    # ---------------------------------------------------------------------
    rel_ds = gdal.Open(input_relative, gdal.GA_ReadOnly)
    if rel_ds is None:
        raise IOError(f"Cannot open input_relative: {input_relative}")

    rel_geoTransform = rel_ds.GetGeoTransform()
    rel_projection   = rel_ds.GetProjection()
    width  = rel_ds.RasterXSize
    height = rel_ds.RasterYSize

    minx = rel_geoTransform[0]
    maxy = rel_geoTransform[3]
    resx = rel_geoTransform[1]
    resy = rel_geoTransform[5]  # typically negative
    maxx = minx + width  * resx
    miny = maxy + height * resy

    # ---------------------------------------------------------------------
    # 2) Clip/resample reference DEM to match same extent/resolution
    # ---------------------------------------------------------------------
    ref_clipped_ds = gdal.Translate(
        '',  # in-memory
        input_ref,
        format='MEM',
        projWin=[minx, maxy, maxx, miny],
        xRes=abs(resx),
        yRes=abs(resy),
        resampleAlg=gdal.GRA_Cubic
    )
    if ref_clipped_ds is None:
        raise IOError(f"Cannot open or clip input_ref: {input_ref}")

    ref_band = ref_clipped_ds.GetRasterBand(1)
    NDV_ref = ref_band.GetNoDataValue()

    # We can read the entire band
    ref_arr = ref_band.ReadAsArray().astype(np.float32)

    # Check if NoData present
    if NDV_ref is not None:
        # If there's ANY NoData in the clipped reference, we skip or handle it
        nd_mask = (ref_arr == NDV_ref)
        if np.any(nd_mask):
            # For this example, let's skip if any ND is found
            print(f"Warning: NoData detected in reference tile. Skipping tile:")
            print(f"  ref: {input_ref}")
            print(f"  rel: {input_relative}")
            return

    # ---------------------------------------------------------------------
    # 3) Read relative DEM array
    # ---------------------------------------------------------------------
    rel_arr = rel_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Confirm shape
    if ref_arr.shape != rel_arr.shape:
        raise ValueError("Shapes of reference and relative DEM do not match after resampling.")

    # ---------------------------------------------------------------------
    # 4) (Optional) Normalize [0..1] => [-1..+1]
    # ---------------------------------------------------------------------
    if do_normalize:
        # This assumes rel_arr is mostly in [0..1].
        rel_arr = 2.0 * (rel_arr - 0.5)  # now in [-1..+1]

    # ---------------------------------------------------------------------
    # 5) Scale the relative DEM
    # ---------------------------------------------------------------------
    rel_scaled = scale_factor * rel_arr

    # ---------------------------------------------------------------------
    # 6) difference = reference - rel_scaled
    # ---------------------------------------------------------------------
    diff_arr = ref_arr - rel_scaled

    # ---------------------------------------------------------------------
    # 7) Fit both polynomial orders => pick best by RMSE
    # ---------------------------------------------------------------------
    h, w = diff_arr.shape
    y_idx, x_idx = np.indices((h, w)).astype(np.float32)

    diff_flat = diff_arr.ravel()
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()

    # Fit order=1
    coeffs1 = fit_polynomial_surface(x_flat, y_flat, diff_flat, order=1)
    diff_poly1 = evaluate_polynomial_surface(x_idx, y_idx, coeffs1, order=1)
    rmse1 = compute_rmse(diff_arr, diff_poly1)

    # Fit order=2
    coeffs2 = fit_polynomial_surface(x_flat, y_flat, diff_flat, order=2)
    diff_poly2 = evaluate_polynomial_surface(x_idx, y_idx, coeffs2, order=2)
    rmse2 = compute_rmse(diff_arr, diff_poly2)

    if rmse2 < rmse1:
        chosen_order = 2
        diff_poly_best = diff_poly2
    else:
        chosen_order = 1
        diff_poly_best = diff_poly1

    final_arr = rel_scaled + diff_poly_best

    # ---------------------------------------------------------------------
    # 8) Rescale final to [ref_min - 1.5, ref_max + 1.5] (example)
    #    using gdal in-memory approach
    # ---------------------------------------------------------------------
    # Create an in-memory dataset for final_arr
    mem_driver = gdal.GetDriverByName('MEM')
    mem_ds = mem_driver.Create('', w, h, 1, gdal.GDT_Float32)
    mem_ds.SetGeoTransform(rel_geoTransform)
    mem_ds.SetProjection(rel_projection)
    mem_band = mem_ds.GetRasterBand(1)
    # Use reference ND if present
    if NDV_ref is not None:
        mem_band.SetNoDataValue(NDV_ref)
    mem_band.WriteArray(final_arr)

    # Compute stats of final
    final_minmax = mem_band.ComputeStatistics(False)  # [min, max, mean, std]
    final_min = final_minmax[0]
    final_max = final_minmax[1]

    # Compute stats of reference tile
    ref_stats = ref_band.ComputeStatistics(False)
    ref_min = ref_stats[0]
    ref_max = ref_stats[1]

    # Decide your offset
    offset_val = 1.5 #0.375
    # Use gdal.Translate to rescale final to [ref_min - offset_val, ref_max + offset_val]
    scaleParams = [[final_min, final_max, ref_min - offset_val, ref_max + offset_val]]

    scaled_mem_ds = gdal.Translate(
        '',  # output to memory
        mem_ds,
        format='MEM',
        scaleParams=scaleParams
    )

    # ---------------------------------------------------------------------
    # 9) Save final to disk with the reference’s ND if present
    # ---------------------------------------------------------------------
    driver_gtiff = gdal.GetDriverByName('GTiff')
    out_ds = driver_gtiff.CreateCopy(output_final, scaled_mem_ds, strict=0)
    out_band = out_ds.GetRasterBand(1)
    if NDV_ref is not None:
        out_band.SetNoDataValue(NDV_ref)
    else:
        # If there's no ND in reference, optionally set your own
        # out_band.SetNoDataValue(-9999)
        pass
    out_band.FlushCache()

    # Print some info
    print(f"Auto-selected polynomial order = {chosen_order}")
    print(f"  RMSE(order=1) = {rmse1:.4f}, RMSE(order=2) = {rmse2:.4f}")
    print(f"Rescaled final DEM from [{final_min:.2f}, {final_max:.2f}] to [{ref_min - offset_val:.2f}, {ref_max + offset_val:.2f}]")

    # Cleanup
    mem_ds = None
    scaled_mem_ds = None
    out_ds = None
    rel_ds = None
    ref_clipped_ds = None


def main():
    if len(sys.argv) < 4:
        print("Usage: python postprocess_poly_auto_scale.py <input_ref> <input_relative> <output_final> "
              "[<scale_factor>] [<do_normalize>]")
        sys.exit(1)

    input_ref      = sys.argv[1]
    input_relative = sys.argv[2]
    output_final   = sys.argv[3]

    # Default parameters
    scale_factor = 1.0
    do_normalize = False

    if len(sys.argv) >= 5:
        scale_factor = float(sys.argv[4])
    if len(sys.argv) >= 6:
        do_normalize = (sys.argv[5].lower() in ['true', '1', 'yes'])

    postprocess(input_ref, input_relative, output_final,
                scale_factor=scale_factor, 
                do_normalize=do_normalize)

    print(f"Done. Output saved to: {output_final}")

if __name__ == "__main__":
    main()

