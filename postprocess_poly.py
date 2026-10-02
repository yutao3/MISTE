#!/usr/bin/env python3
"""
Script to combine a low-frequency reference DEM (absolute) with a high-frequency relative DEM,
using a polynomial surface fit to align the large-scale slope/offset. 
This version includes:
  - Optional normalization of the relative DEM from [0..1] to [-1..1] (or any desired shift).
  - A scale factor applied BEFORE we compute the polynomial difference.

Usage:
    python postprocess_poly.py <input_ref> <input_relative> <output_final> \
                               [<poly_order>] [<scale_factor>] [<do_normalize>]

Examples:
    1) No scaling, no normalization (order=1):
       python postprocess_poly.py ref_dem.tif rel_dem.tif fused_dem.tif

    2) Polynomial order=1, scaling factor=5.0, no normalization:
       python postprocess_poly.py ref_dem.tif rel_dem.tif fused_dem.tif 1 5.0

    3) Polynomial order=2, scale=10, with normalization:
       python postprocess_poly.py ref_dem.tif rel_dem.tif fused_dem.tif 2 10.0 True
"""

import sys
import numpy as np
from osgeo import gdal

def fit_polynomial_surface(x, y, z, order=1):
    """
    Fit a 2D polynomial of given 'order' to data z = f(x, y),
    returning coefficients in a flattened form.
    
    For order=1, fits a plane: z = c0 + c1*x + c2*y.
    For order=2, fits z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2.
    
    x, y, z should be 1D arrays of the same length.
    """
    if order < 1 or order > 2:
        raise ValueError("Currently only orders 1 or 2 are implemented in this example.")

    # Flatten inputs
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # Construct design matrix
    if order == 1:
        # Plane: a + b*x + c*y
        M = np.column_stack([np.ones_like(x), x, y])
    else:
        # Quadratic: a + b*x + c*y + d*x^2 + e*x*y + f*y^2
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

def postprocess(input_ref, input_relative, output_final, 
                           poly_order=1, scale_factor=1.0, do_normalize=False):
    """
    Main workflow:
      1. Align/clip reference to match the relative DEM's extent/resolution.
      2. Optionally normalize the relative DEM from [0..1] to [-1..+1] (or another shift).
      3. Scale the relative DEM by scale_factor.
      4. difference = reference - scaled_relative
      5. Fit polynomial surface to difference
      6. final = scaled_relative + polynomial_surface
      7. Write out final as GeoTIFF with the relative DEM's georef.
    """
    # ---------------------------------------------------------------------
    # 1) Open the relative DEM to get its geotransform & size
    # ---------------------------------------------------------------------
    rel_ds = gdal.Open(input_relative)
    if rel_ds is None:
        raise IOError(f"Cannot open input_relative: {input_relative}")

    # We'll preserve these for the output
    rel_geoTransform = rel_ds.GetGeoTransform()
    rel_projection   = rel_ds.GetProjection()
    width  = rel_ds.RasterXSize
    height = rel_ds.RasterYSize

    # Get bounding box from relative DEM
    minx = rel_geoTransform[0]
    maxy = rel_geoTransform[3]
    resx = rel_geoTransform[1]
    resy = rel_geoTransform[5]  # typically negative
    maxx = minx + width  * resx
    miny = maxy + height * resy

    # ---------------------------------------------------------------------
    # 2) Resample/clip reference DEM to match the same extent & resolution
    # ---------------------------------------------------------------------
    ref_clipped_ds = gdal.Translate(
        '',  # output to memory
        input_ref,
        format='MEM',
        projWin=[minx, maxy, maxx, miny],  # bounding box
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
    
    # Confirm dimensions match
    if ref_arr.shape != rel_arr.shape:
        raise ValueError("Shapes of reference and relative DEM do not match after resampling.")

    # ---------------------------------------------------------------------
    # 4) Optional: Normalize the relative DEM (e.g., from [0..1] to [-1..+1])
    # ---------------------------------------------------------------------
    # A simple approach if you know rel_arr is in [0..1]:
    # rel_norm = 2*(rel - 0.5) => [-1..+1].
    # Then you can scale it. 
    if do_normalize:
        # Transform from [0..1] to [-1..1], assuming original is mostly in [0..1].
        # If your relative DEM is guaranteed 0..1, you can clamp or adapt as needed.
        rel_arr = 2.0 * (rel_arr - 0.5)  # Now in [-1..+1]
    
    # ---------------------------------------------------------------------
    # 5) Scale the relative DEM (if scale_factor != 1)
    # ---------------------------------------------------------------------
    rel_scaled = scale_factor * rel_arr

    # ---------------------------------------------------------------------
    # 6) difference = reference - scaled_relative
    # ---------------------------------------------------------------------
    diff_arr = ref_arr - rel_scaled

    # ---------------------------------------------------------------------
    # 7) Fit polynomial surface to the difference
    # ---------------------------------------------------------------------
    h, w = diff_arr.shape
    y_idx, x_idx = np.indices((h, w)).astype(np.float32)

    diff_flat = diff_arr.ravel()

    coeffs = fit_polynomial_surface(x_idx, y_idx, diff_flat, order=poly_order)

    # ---------------------------------------------------------------------
    # 8) Evaluate polynomial surface to get the large-scale difference
    # ---------------------------------------------------------------------
    diff_poly = evaluate_polynomial_surface(x_idx, y_idx, coeffs, order=poly_order)

    # ---------------------------------------------------------------------
    # 9) final = scaled_relative + polynomial_surface
    # ---------------------------------------------------------------------
    final_arr = rel_scaled + diff_poly

    # ---------------------------------------------------------------------
    # 10) Write out final DEM with the same georeferencing as input_relative
    # ---------------------------------------------------------------------
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_final,
        width,
        height,
        1,
        gdal.GDT_Float32
    )
    out_ds.SetGeoTransform(rel_geoTransform)
    out_ds.SetProjection(rel_projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(final_arr)
    out_band.SetNoDataValue(-9999)  # optional
    out_band.FlushCache()

    # Clean up
    rel_ds = None
    ref_clipped_ds = None
    out_ds = None

def main():
    if len(sys.argv) < 4:
        print("Usage: python postprocess_poly.py <input_ref> <input_relative> <output_final> "
              "[<poly_order>] [<scale_factor>] [<do_normalize>]")
        sys.exit(1)

    input_ref      = sys.argv[1]
    input_relative = sys.argv[2]
    output_final   = sys.argv[3]

    # Default polynomial order
    poly_order = 1
    # Default scale_factor
    scale_factor = 1.5
    # Default do_normalize
    do_normalize = False

    if len(sys.argv) >= 5:
        poly_order = int(sys.argv[4])
    if len(sys.argv) >= 6:
        scale_factor = float(sys.argv[5])
    if len(sys.argv) >= 7:
        do_normalize = (sys.argv[6].lower() in ['true', '1', 'yes'])

    postprocess(input_ref, input_relative, output_final, 
                           poly_order=poly_order, 
                           scale_factor=scale_factor, 
                           do_normalize=do_normalize)
    print(f"Done. Output saved to: {output_final}")

if __name__ == "__main__":
    main()

