#!/usr/bin/env python

from osgeo import gdal
from osgeo import gdal_array
import numpy as np
import sys

def postprocess(input_ref, input_relative, output_final):
    # Read input files
    data = gdal.Open(input_relative)
    refdata = gdal.Open(input_ref)
    
    # Preserve geo-information for output
    geoTransform = data.GetGeoTransform()
    projection = data.GetProjection()

    # Perform resampling and clip
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    resx = geoTransform[1]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize

    #refdata_resampled = gdal.Warp('', refdata, format='MEM', resampleAlg=gdal.GRA_CubicSpline, xRes=resx, yRes=resx, outputBounds=[minx, miny, maxx, maxy])                              
    refdata_resampled = gdal.Translate('', refdata, format='MEM', resampleAlg=gdal.GRA_CubicSpline, xRes=resx, yRes=resx, projWin=[minx, maxy, maxx, miny])
    
    # Check for NoData values
    srcband = refdata_resampled.GetRasterBand(1)
    NDV = srcband.GetNoDataValue()
    srcband_check = srcband.ReadAsArray()

    if (srcband_check == NDV).any():
        print("Warning: NoData value detected in the reference DTM, {}, for this ORI-DTM inference tiles, {}. Predicted DTM for this tile is removed. This may result in gaps in the final DTM.".format(input_ref, input_relative))
        return

    # Scaling input_relative
    stats = srcband.GetStatistics(True, True)
    stat_min = stats[0]
    stat_max = stats[1]

    data_scaled = gdal.Translate('', data, format='MEM', scaleParams=[[0, 1, stat_min - 1.25, stat_max + 1.25]])
    # use 0.175 for HiRISE, 1.75 for HRSC

    # Difference calculation
    diff = gdal_array.OpenArray(data_scaled.GetRasterBand(1).ReadAsArray() - refdata_resampled.GetRasterBand(1).ReadAsArray())

    # Resampling diff
    diff_resampled_down = gdal.Translate('', diff, format='MEM', width=round(diff.RasterXSize * 0.05), height=round(diff.RasterYSize * 0.05), resampleAlg=gdal.GRA_Average)
    diff_resampled_up = gdal.Translate('', diff_resampled_down, format='MEM', width=diff.RasterXSize, height=diff.RasterYSize, resampleAlg=gdal.GRA_CubicSpline)

    # Final calculation
    final_arr = data_scaled.GetRasterBand(1).ReadAsArray() - diff_resampled_up.GetRasterBand(1).ReadAsArray()
    final = gdal_array.OpenArray(final_arr)

    # Save output_final
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(output_final, final)

    # Set georeferencing information for the output file
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds.SetProjection(projection)

    # Clean up
    refdata = None
    data = None
    refdata_resampled = None
    data_scaled = None
    diff = None
    diff_resampled_down = None
    diff_resampled_up = None
    final = None
    dst_ds = None


if __name__ == "__main__":
    postprocess(sys.argv[1],sys.argv[2], sys.argv[3])
