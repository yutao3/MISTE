#!/usr/bin/env python

import numpy as np
from osgeo import gdal
from osgeo import osr
import cv2
import os
import sys

# SMALL IMAGE USE overlap = 150, LARGE IMAGE USE 120, USE 350 TO FULLY ELIMINATE TILING ARTEFACT
def make_tiles(ori_filepath, img_folder, geo_folder, overlap = 280, patch_width = 640, patch_height = 480):    
    if os.path.isfile(ori_filepath):
        ori_ds = gdal.Open(ori_filepath)
        im_height = ori_ds.RasterYSize
        im_width = ori_ds.RasterXSize

        # crop
        patch_no = 0
        for i in np.arange(0, im_height-patch_height, patch_height - overlap):
            for j in np.arange(0,im_width-patch_width,patch_width - overlap):
                # crop ORI
                crop_ds = gdal.Translate('temp.tif', ori_ds, srcWin = [j, i, patch_width, patch_height])              
                crop_ori = crop_ds.ReadAsArray()
                
                if np.sum(crop_ori >= 20) <= 0.85 * crop_ori.size:
                    os.system('rm temp.tif')
                    continue
                elif ((crop_ori.size - np.count_nonzero(crop_ori)) >= 3000):
                    os.system('rm temp.tif')
                    continue
                else:
                    # save image into img_folder
                    patch_no= patch_no + 1           
                    ori_dirname, ori_filename = os.path.split(ori_filepath)
                    ori_prefix, ori_extension = os.path.splitext(ori_filename)
                    crop_ori_img = img_folder + '/' + ori_prefix + '_' + str(patch_no) + ori_extension                   
                    crop_ori_threeband = cv2.cvtColor(crop_ori, cv2.COLOR_GRAY2RGB)
                    cv2.imwrite(crop_ori_img,crop_ori_threeband)                    
                    # move temp.tif to geo_folder
                    crop_ori_geo = geo_folder + '/' + ori_prefix + '_' + str(patch_no) + ori_extension
                    os.system('mv temp.tif %s' %crop_ori_geo )
                            
        print('patch number : %d' % patch_no)

if __name__ == "__main__":
    make_tiles(sys.argv[1],sys.argv[2], sys.argv[3])

