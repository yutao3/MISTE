#!/usr/bin/env python3
"""prepare_geotiff_lroc_nac.py  –  Sequentially process an LROC NAC EDR into an 8‑bit GeoTIFF

Usage
-----
    python prepare_geotiff_lroc_nac.py input.IMG input_projection.map

This script runs the following steps in order, verifying output at each stage:

1. **Import to ISIS**:
   lronac2isis from=INPUT.IMG to=INPUT.lev0.cub
2. **SPICE init**:
   spiceinit from=INPUT.lev0.cub web=yes
3. **Radiometric calibration**:
   lronaccal from=INPUT.lev0.cub to=INPUT.lev1.cub
4. **Destriping**:
   lronacecho from=INPUT.lev1.cub to=INPUT.lev2.cub
5. **Map‐project**:
   cam2map from=INPUT.lev2.cub map=PROJ.map to=INPUT.lev3.cub
6. **Convert to GeoTIFF**:
   gdal_translate -of GTiff -b 1 -tr 1 1 INPUT.lev3.cub INPUT.tif
7. **Scale to 8‑bit**:
   - Compute min/max via GDAL
   - gdal_translate -of GTiff -ot Byte -scale MIN MAX 1 255 INPUT.tif INPUT_8BIT.tif

Intermediate files are removed at the end, leaving only the final 8‑bit TIFF.
"""

import sys
import subprocess
from pathlib import Path

try:
    from osgeo import gdal
except ImportError:
    gdal = None


def run(cmd: str, step: str) -> None:
    print(f"[Step] {step}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode:
        print(f"Error: '{step}' failed (code {res.returncode})")
        sys.exit(1)


def check_file(p: Path, step: str) -> None:
    if not p.exists():
        print(f"Error: Expected output missing after '{step}': {p.name}")
        sys.exit(1)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    img = Path(sys.argv[1])
    proj = Path(sys.argv[2])
    if not img.exists():
        print(f"Input file not found: {img}")
        sys.exit(1)
    if not proj.exists():
        print(f"Projection file not found: {proj}")
        sys.exit(1)

    base = img.stem
    outdir = img.parent

    # File names
    lev0 = outdir / f"{base}.lev0.cub"
    lev1 = outdir / f"{base}.lev1.cub"
    lev2 = outdir / f"{base}.lev2.cub"
    lev3 = outdir / f"{base}.lev3.cub"
    tif   = outdir / f"{base}.tif"
    tif8  = outdir / f"{base}_8BIT.tif"

    # 1: Import
    run(f"lronac2isis from={img} to={lev0}", "Import to ISIS")
    check_file(lev0, "Import to ISIS")

    # 2: SPICE init
    run(f"spiceinit from={lev0} web=yes", "SPICE initialization")
    check_file(lev0, "SPICE initialization")

    # 3: Radiometric calibration
    run(f"lronaccal from={lev0} to={lev1}", "Radiometric calibration")
    check_file(lev1, "Radiometric calibration")

    # 4: Destriping
    run(f"lronacecho from={lev1} to={lev2}", "Destriping (lronacecho)")
    check_file(lev2, "Destriping (lronacecho)")

    # 5: Map projection
    run(f"cam2map from={lev2} map={proj} to={lev3}", "Map projection (cam2map)")
    check_file(lev3, "Map projection (cam2map)")

    # 6: Convert to GeoTIFF
    run(f"gdal_translate -of GTiff -b 1 {lev3} {tif}", "Convert to GeoTIFF")
    check_file(tif, "Convert to GeoTIFF")

    # 7: Scale to 8-bit
    # Determine min/max via GDAL or gdalinfo
    if gdal:
        ds = gdal.Open(str(tif))
        band = ds.GetRasterBand(1)
        stats = band.GetStatistics(False, True)
        mn, mx = stats[0], stats[1]
    else:
        # Fallback to gdalinfo
        info = subprocess.check_output(["gdalinfo", "-mm", str(tif)]).decode()
        mn = mx = None
        for line in info.splitlines():
            if "Computed Min/Max=" in line:
                try:
                    # Example: '  Computed Min/Max=-0.006,0.089'
                    parts = line.split('=')[-1].strip().split(',')
                    mn = float(parts[0].strip())
                    mx = float(parts[1].strip())
                    break
                except (ValueError, IndexError):
                    print("Error: Unable to parse min/max values from line:")
                    print(line)
                    sys.exit(1)
        if mn is None or mx is None:
            print("Error: Cannot determine min/max for scaling")
            sys.exit(1)

    run(f"gdal_translate -of GTiff -ot Byte -scale {mn} {mx} 1 255 {tif} {tif8}", "Scale to 8-bit TIFF")
    check_file(tif8, "Scale to 8-bit TIFF")

    # Cleanup
    for f in (lev0, lev1, lev2, lev3, tif):
        try:
            f.unlink()
        except Exception:
            pass

    print(f"Finished: final product → {tif8}")


if __name__ == "__main__":
    main()

