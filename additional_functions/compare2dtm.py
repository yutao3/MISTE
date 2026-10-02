import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.signal import fftconvolve
import math
import os
import numpy.ma as ma
#from PIL import ImageChops
#import math, operator
import sys

# Our implementation of Kirk, RS 13(17) 3511, 2021. 

def readgeotiff(filepath):
	'''
	read geotif file
	'''
	ds = gdal.Open(filepath)
	NDV = ds.GetRasterBand(1).GetNoDataValue()
	#xsize = ds.RasterXSize
	#ysize = ds.RasterYSize
	#GeoT = ds.GetGeoTransform()
	#Projection = osr.SpatialReference()
	#Projection.ImportFromWkt(ds.GetProjectionRef())
	#DataType = ds.GetRasterBand(1).DataType
	#DataType = gdal.GetDataTypeName(DataType)
	data = ds.ReadAsArray()
	return data, NDV #, xsize, ysize, GeoT, Projection, DataType

def getlayerextent(filepath):
	ds = gdal.Open(filepath)
	xsize = ds.RasterXSize
	ysize = ds.RasterYSize
	GeoT = ds.GetGeoTransform()
	buff = 0
	if GeoT is not None:
		GeoXUL = GeoT[0] -buff
		GeoYUL = GeoT[3] +buff
		GeoXLR = GeoT[0] + GeoT[1] * xsize + GeoT[2] * ysize + buff
		GeoYLR = GeoT[3] + GeoT[4] * xsize + GeoT[5] * ysize - buff
	# return the four corners coordinates
	return GeoXUL, GeoYUL, GeoXLR, GeoYLR

def boxcar(A, nodata, window_size = 3):
	if nodata > 0:
		mask = (A>=nodata)
	else:
		mask = (A<=nodata)
	K = np.ones((window_size, window_size),dtype=int)

	out = np.round(fftconvolve(np.where(mask,0,A), K, mode="same")/fftconvolve(~mask,K, mode="same"), 2)
	out[mask] = nodata

	return np.ma.masked_array(out, mask=(out == nodata)) 


def compare2dtm(refdtmfile, tardtmfile, maxfilt_width = 65):
	'''
	compare two dtms
	'''
	# get xsize and ysize of the target dtm
	tards = gdal.Open(tardtmfile)
	GeoT = tards.GetGeoTransform()
	xsize = abs(GeoT[1])
	ysize = abs(GeoT[5])

	# gdal_tranlsate the refdtm to the xsize and ysize
	refds = gdal.Open(refdtmfile)
	file, ext = os.path.splitext(os.path.basename(refdtmfile))
	refdtmfile = file + '-downsampled' + ext
	refds = gdal.Translate(refdtmfile, refds, xRes=xsize, yRes=ysize, resampleAlg='cubic')

	refGeoXUL, refGeoYUL, refGeoXLR, refGeoYLR = getlayerextent(refdtmfile)
	tarGeoXUL, tarGeoYUL, tarGeoXLR, tarGeoYLR = getlayerextent(tardtmfile)

	# calculate the inner overlap area
	XUL = np.max((refGeoXUL, tarGeoXUL))
	YUL = np.min((refGeoYUL, tarGeoYUL))
	XLR = np.min((refGeoXLR, tarGeoXLR))
	YLR = np.max((refGeoYLR, tarGeoYLR))

	# call gdal_translate to cut the two image to the new area
	file, ext = os.path.splitext(os.path.basename(tardtmfile))
	croptardtmfile = file + '-crop' + ext
	file, ext = os.path.splitext(os.path.basename(refdtmfile))
	croprefdtmfile = file + '-crop' + ext
	
	#refds = cropgeotiff(refdtmfile, croprefdtmfile, [XUL, YUL, XLR, YLR])
	#tards = cropgeotiff(tardtmfile, croptardtmfile, [XUL, YUL, XLR, YLR])
	refds = gdal.Translate(croprefdtmfile, refds, projWin=[XUL, YUL, XLR, YLR])
	tards = gdal.Translate(croptardtmfile, tards, projWin=[XUL, YUL, XLR, YLR])

	# read files again
	refdtm = refds.ReadAsArray()
	refnodata = refds.GetRasterBand(1).GetNoDataValue()

	tardtm = tards.ReadAsArray()
	tarnodata = tards.GetRasterBand(1).GetNoDataValue()
	#refdtm, refnodata = readgeotiff(croprefdtmfile)
	#tardtm, tarnodata = readgeotiff(croptardtmfile)

	maxfilt_width = int(maxfilt_width)
	# boxcar filter -lowpass fitler with filter width 1*1, 3*3, 5*5
	width_arr = np.arange(1, maxfilt_width,2)
	rmse_arr = np.zeros(width_arr.shape)
	ssim_arr = np.zeros(width_arr.shape)
	for k, width in enumerate(width_arr):
		# boxcar filter -lowpass fitler 
		refdtm_filter = boxcar(refdtm, refnodata, width)

		# save every filtered images
		file, ext = os.path.splitext(os.path.basename(croprefdtmfile))
		filtfilename = file + '-boxcar-' +str(width) + ext
		driver = gdal.GetDriverByName('GTiff')
		fltds = driver.CreateCopy(filtfilename, refds, 0)
		fltds.GetRasterBand(1).WriteArray(refdtm_filter)
		fltds.GetRasterBand(1).SetNoDataValue(refnodata)

		# mask nodata area
		tardtm = ma.masked_where(tardtm == tarnodata, tardtm)

		masknew = (refdtm_filter.mask + tardtm.mask )

		refdtm_filter = np.ma.masked_array(refdtm_filter, mask=masknew) 
		tardtm = np.ma.masked_array(tardtm, mask=masknew) 

		# cut into blocks
		im_height, im_width = tardtm.shape
		patch_height = im_height//10
		patch_width  = im_width//10
		
		#for very small overlaps need to manually set a number like 200-600.
		#patch_height = 100
		#patch_width  = 100
		overlap = 0

		rmselist = []
		ssimlist = []
		for i in np.arange(0, im_height-patch_height, patch_height - overlap):
			for j in np.arange(0,im_width-patch_width,patch_width - overlap):

				if np.any(masknew[i:i+patch_height,j:j+patch_width]):
					continue

				refdtm_patch = refdtm_filter[i:i+patch_height,j:j+patch_width]
				tardtm_patch = tardtm[i:i+patch_height,j:j+patch_width]
				#new_refdtm_patch = refdtm_patch - (refdtm_patch - tardtm_patch).mean()

				#rmselist.append(mean_squared_error(new_refdtm_patch, tardtm_patch))
				#ssimlist.append(ssim(new_refdtm_patch, tardtm_patch, data_range = tardtm_patch.max()-tardtm_patch.min()))
				rmselist.append(math.sqrt(mean_squared_error(refdtm_patch, tardtm_patch)))
				ssimlist.append(ssim(refdtm_patch, tardtm_patch, data_range = tardtm_patch.max()-tardtm_patch.min()))
		
		# calculate rmse and ssim
		rmse_arr[k] = np.mean(rmselist)
		ssim_arr[k] = np.mean(ssimlist)
	
	table = np.vstack((width_arr, rmse_arr, ssim_arr)).T
	np.savetxt('result.txt',table, fmt = '%.4f')

	# plot the rmse and ssim with regard to the width array
	plt.figure(figsize=(12,6))
	plt.subplot(1,2,1)
	plt.xlabel('Smoothing Filter Width (pixels)')
	plt.ylabel('RMSE (metres)')
	plt.plot(width_arr, rmse_arr)
	plt.plot(width_arr[np.argmin(rmse_arr)], np.min(rmse_arr), 'x')
	print(width_arr[np.argmin(rmse_arr)])
	plt.subplot(1,2,2)
	plt.xlabel('Smoothing Filter Width (pixels)')
	plt.ylabel('Mean SSIM (0-1)')
	plt.plot(width_arr, ssim_arr)
	plt.plot(width_arr[np.argmax(ssim_arr)], np.max(ssim_arr), 'x')
	print(width_arr[np.argmax(ssim_arr)])

	plt.savefig('result.png')



if __name__ == "__main__":

	compare2dtm(sys.argv[1],sys.argv[2], sys.argv[3])
	
	# ref target filter-size-max
