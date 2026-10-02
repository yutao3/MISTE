#!/usr/bin/env python3
"""
Script to combine a reference DEM (absolute) with a high-frequency relative DEM,
using a polynomial surface fit to align large-scale slope/offset.

Features:
  - Optional normalization of the relative DEM from [0..1] to [-1..+1].
  - A scale factor applied BEFORE computing difference with the reference.
  - Automatic polynomial order selection (1 vs. 2) based on residual error.

Usage:
    python postprocess_poly_auto.py <input_ref> <input_relative> <output_final> \
                                    [<scale_factor>] [<do_normalize>]

Examples:
    1) No scaling, no normalization, auto order selection:
       python postprocess_poly_auto.py ref_dem.tif rel_dem.tif fused_dem.tif

    2) Scale=5.0, no normalization, auto order selection:
       python postprocess_poly_auto.py ref_dem.tif rel_dem.tif fused_dem.tif 5.0

    3) Scale=10, with normalization, auto order selection:
       python postprocess_poly_auto.py ref_dem.tif rel_dem.tif fused_dem.tif 10 True
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

    # Flatten inputs
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # Construct design matrix
    if order == 1:
        # plane: a + b*x + c*y
        M = np.column_stack([np.ones_like(x), x, y])
    else:
        # quadratic: a + b*x + c*y + d*x^2 + e*x*y + f*y^2
        M = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])

    # Solve least squares: M * coeffs = z
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
    rmse_val = np.sqrt(mse)
    return rmse_val

def postprocess(input_ref, input_relative, output_final, 
                                scale_factor=1.0, do_normalize=False):
    """
    Main workflow:
      1. Align/clip reference to match the relative DEM's extent/resolution.
      2. (Optional) Normalize the relative DEM from [0..1] to [-1..+1].
      3. Scale the relative DEM.
      4. difference = reference - (scaled_relative).
      5. Fit both polynomial order=1 and order=2. Compare residuals (RMSE).
      6. Choose the better order automatically.
      7. Evaluate chosen polynomial, final = scaled_relative + polynomial_surface.
      8. Write out final.

    This approach ensures minimal large-scale residual.
    """
    # ---------------------------------------------------------------------
    # 1) Open the relative DEM to get its geotransform & size
    # ---------------------------------------------------------------------
    rel_ds = gdal.Open(input_relative)
    if rel_ds is None:
        raise IOError(f"Cannot open input_relative: {input_relative}")

    # We'll preserve these for output
    rel_geoTransform = rel_ds.GetGeoTransform()
    rel_projection   = rel_ds.GetProjection()
    width  = rel_ds.RasterXSize
    height = rel_ds.RasterYSize

    # Derive bounding box from relative DEM
    minx = rel_geoTransform[0]
    maxy = rel_geoTransform[3]
    resx = rel_geoTransform[1]
    resy = rel_geoTransform[5]  # typically negative
    maxx = minx + width  * resx
    miny = maxy + height * resy

    # ---------------------------------------------------------------------
    # 2) Resample/clip reference DEM to match the relative DEM's extent/res
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

    # ---------------------------------------------------------------------
    # 3) Read arrays from both datasets
    # ---------------------------------------------------------------------
    ref_arr = ref_clipped_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    rel_arr = rel_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Confirm shape match
    if ref_arr.shape != rel_arr.shape:
        raise ValueError("Shapes of reference and relative DEM do not match after resampling.")

    # ---------------------------------------------------------------------
    # 4) (Optional) Normalize the relative DEM from [0..1] to [-1..+1]
    # ---------------------------------------------------------------------
    if do_normalize:
        # Simple transform, assuming rel_arr is roughly [0..1].
        rel_arr = 2.0 * (rel_arr - 0.5)  # now in [-1..+1]

    # ---------------------------------------------------------------------
    # 5) Scale the relative DEM
    # ---------------------------------------------------------------------
    rel_scaled = scale_factor * rel_arr

    # ---------------------------------------------------------------------
    # 6) Compute difference = reference - rel_scaled
    # ---------------------------------------------------------------------
    diff_arr = ref_arr - rel_scaled

    # ---------------------------------------------------------------------
    # 7) Fit both polynomial orders (1 and 2). Compare residuals.
    # ---------------------------------------------------------------------
    # We'll generate x, y pixel coords
    h, w = diff_arr.shape
    y_idx, x_idx = np.indices((h, w)).astype(np.float32)

    diff_flat = diff_arr.ravel()
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()

    # ---- Fit order=1
    coeffs1 = fit_polynomial_surface(x_flat, y_flat, diff_flat, order=1)
    diff_poly1 = evaluate_polynomial_surface(x_idx, y_idx, coeffs1, order=1)
    # RMSE for order=1
    rmse1 = compute_rmse(diff_arr, diff_poly1)

    # ---- Fit order=2
    coeffs2 = fit_polynomial_surface(x_flat, y_flat, diff_flat, order=2)
    diff_poly2 = evaluate_polynomial_surface(x_idx, y_idx, coeffs2, order=2)
    # RMSE for order=2
    rmse2 = compute_rmse(diff_arr, diff_poly2)

    # Decide which order to use
    if rmse2 < rmse1:
        chosen_order = 2
        coeffs_best = coeffs2
        diff_poly_best = diff_poly2
    else:
        chosen_order = 1
        coeffs_best = coeffs1
        diff_poly_best = diff_poly1

    # ---------------------------------------------------------------------
    # 8) Create final DEM = rel_scaled + diff_poly_best
    # ---------------------------------------------------------------------
    final_arr = rel_scaled + diff_poly_best

    # ---------------------------------------------------------------------
    # 9) Write out final
    # ---------------------------------------------------------------------
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_final, width, height, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(rel_geoTransform)
    out_ds.SetProjection(rel_projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(final_arr)
    out_band.SetNoDataValue(-9999)  # optional
    out_band.FlushCache()

    # Print a small log about the chosen order and RMSE
    # print(f"Auto-selected polynomial order = {chosen_order}")
    # print(f"  RMSE(order=1) = {rmse1:.4f}")
    # print(f"  RMSE(order=2) = {rmse2:.4f}")

    # Cleanup
    rel_ds = None
    ref_clipped_ds = None
    out_ds = None


def main():
    if len(sys.argv) < 4:
        print("Usage: python postprocess.py <input_ref> <input_relative> <output_final> "
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

    # Run auto postprocess
    postprocess(input_ref, input_relative, output_final,
                                scale_factor=scale_factor,
                                do_normalize=do_normalize)
    print(f"Done. Output saved to: {output_final}")

if __name__ == "__main__":
    main()

