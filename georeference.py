#!/usr/bin/env python
import numpy as np
from osgeo import gdal
from osgeo import osr
import os
import sys

# Function to read the original file's projection:
def GetGeoInfo(SourceDS):
    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    DataType = SourceDS.GetRasterBand(1).DataType
    DataType = gdal.GetDataTypeName(DataType)
    return NDV, xsize, ysize, GeoT, Projection, DataType

# Function to write a new file.
def CreateGeoTiff(Name, Array, driver, NDV,
                  xsize, ysize, GeoT, Projection, DataType):
    if DataType == 'Float32':
        DataType = gdal.GDT_Float32
    if DataType == 'Byte':
        DataType = gdal.GDT_Byte
    NewFileName = Name+'.tif'
    # Set nans to the original No Data Value
    Array[np.isnan(Array)] = NDV
    # Set up the dataset
    DataSet = driver.Create( NewFileName, xsize, ysize, 1, DataType )
    # the '1' is for band 1.
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection( Projection.ExportToWkt() )
    # Write the array
    DataSet.GetRasterBand(1).WriteArray( Array )
    DataSet.GetRasterBand(1).SetNoDataValue(NDV)
    return NewFileName


def georeference(input_filepath, geoheader_filepath, output_filepath):
    # read in DTM file
    driver = gdal.Open(input_filepath)
    dtm = driver.ReadAsArray()
    # dtm = dtm.astype(np.float32)
    # normalize the dtm to be in range 0.0 to 1.0 removed as it will be normalised in the inference code.
    #dtm = (dtm - np.min(dtm)) / (np.max(dtm) - np.min(dtm))
    
    # read in the geoinfomration
    driver = gdal.Open(geoheader_filepath)
    NDV, xsize, ysize, GeoT, Projection, DataType = GetGeoInfo(driver)
    
    #change geoinformation
    GeoT_dtm = (GeoT[0], GeoT[1], 0.0, GeoT[3], 0.0, GeoT[5])
    DataType = 'Float32'
    NDV = -3.40282265508890445e+38
    
    # Set up the GTiff driver
    new_driver = gdal.GetDriverByName('GTiff')
    # Now turn the array into a GTiff.
    output_filepath, extension = os.path.splitext(output_filepath)
    NewFileName = CreateGeoTiff(output_filepath, dtm, new_driver, NDV, xsize, ysize, GeoT_dtm, Projection, DataType)


if __name__ == "__main__":
    georeference(sys.argv[1],sys.argv[2], sys.argv[3])
