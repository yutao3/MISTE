#!/usr/bin/env python

import argparse
import os
import glob
import shutil
import time
import subprocess
from make_tiles import make_tiles
from inference import main as inference_main
from georeference import georeference
from postprocess import postprocess
from osgeo import gdal
from natsort import natsorted

def validate_geotiff(tiff1, tiff2):
    ds1 = gdal.Open(tiff1)
    ds2 = gdal.Open(tiff2)

    check_res = True

    if ds1 is None or ds2 is None:
        print ('Error: Cannot open the input image or reference DTM with gdal!')
        check_res = False

    # To verify the georeference information and the ext
    projection1 = ds1.GetProjection()
    geotransform1 = ds1.GetGeoTransform()
    if projection1 is None or geotransform1 is None:
        raise ValueError('Error: input image is NOT a geotiff or projection info missing!')
        check_res = False
        
    projection2 = ds2.GetProjection()
    geotransform2 = ds2.GetGeoTransform()
    if projection2 is None or geotransform2 is None:
        raise ValueError('Error: reference DTM is NOT a geotiff or projection info missing!')
        check_res = False
        
    
    input_minx = geotransform1[0] 
    input_miny = geotransform1[3] + ds1.RasterYSize * geotransform1[5]
    input_maxx = geotransform1[0] + ds1.RasterXSize * geotransform1[1]
    input_maxy = geotransform1[3]  
    
    ref_minx = geotransform2[0]
    ref_miny = geotransform2[3] + ds2.RasterYSize * geotransform2[5]
    ref_maxx = geotransform2[0] + ds2.RasterXSize * geotransform2[1]
    ref_maxy = geotransform2[3]  

    # modify here to make the extent coverage check flexible to a few missing pixels which will be handled later on in postprocessing. Temporarily set as 100 here.    
#    if ((input_minx < ref_minx-100) or (input_miny < ref_miny-100) or 
#         (input_maxx > ref_maxx+100) or (input_maxy > ref_maxy+100)):
#        raise ValueError('Error: Reference image does not cover the extent of input image!')
#        check_res = False
#    else:
#        print('Extent check passed: reference DTM mostly covers the extent of input image.')
    
    return check_res

def full_chain(input_path, ref_path, output_path, asp_path, model_id, weights_path, tmp_path):
    
    execution_times = {}
    dirs = [f'{tmp_path}/crop', f'{tmp_path}/geoheader', f'{tmp_path}/relative', f'{tmp_path}/final']
    print('Tmp working directories created.')
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # perform a check on the input and reference images
    if not validate_geotiff(input_path, ref_path):
        print("Geotiff files are not valid or the extent of the reference geotiff image doesn't cover the extent of the input geotiff images.")
        return
    start_time = time.time()
    print('Creating image tiles ...')
    make_tiles(input_path, f'{tmp_path}/crop', f'{tmp_path}/geoheader')
    print('Image tiles created for inference.')
    execution_times['tile_creation'] = time.time() - start_time
    
    start_time = time.time()
    subprocess.run([f'python inference.py --model {model_id} --weights {weights_path} --data {tmp_path}/crop/'], shell=True)
    print('DTM inference completed for all image tiles.')
    execution_times['DTM_inference'] = time.time() - start_time

    start_time = time.time()
    for i in glob.glob(f'{tmp_path}/crop/*_result.tif'):
        base = os.path.basename(i)[:-11]
        georeference(i, f'{tmp_path}/geoheader/{base}.tif', f'{tmp_path}/relative/{base}_relative.tif')
    print('DTM tiles have been georeferenced.')
    execution_times['DTM_georeferencing'] = time.time() - start_time

    start_time = time.time()
    for i in glob.glob(f'{tmp_path}/relative/*.tif'):
        base = os.path.basename(i)[:-12]
        postprocess(ref_path, i, f'{tmp_path}/final/{base}_final.tif')
    print('DTM tiles have been rescaled according to the reference DTM.')
    execution_times['DTM_rescaling'] = time.time() - start_time

    with open(f'{tmp_path}/all.txt', 'w') as f:
        for i in natsorted(glob.glob(f'{tmp_path}/final/*.tif')):
            f.write(f'{i}\n')

    subprocess.run(f'split -l 200 {tmp_path}/all.txt {tmp_path}/list', shell=True)
    print('A list of DTM tiles has been created and splitted for the ASP DTM mosaicing process.')

    start_time = time.time()
    for i in glob.glob(f'{tmp_path}/list*'):
        subprocess.run(f'{asp_path}/dem_mosaic -l {i} -o {i}-mosaic --erode-length 12 --threads 8', shell=True)

    subprocess.run(f'{asp_path}/dem_mosaic {tmp_path}/list*.tif -o {tmp_path}/final --threads 8', shell=True)
    print('DTM tiles have been mosaiced. Cleaning up ...')
    execution_times['ASP_mosaicing'] = time.time() - start_time

    shutil.move(f'{tmp_path}/final-tile-0.tif', output_path)
    shutil.rmtree(f'{tmp_path}')
    print('All done.')
    
    # Report the execution times
    total_time = sum(execution_times.values())
    print(f'Total processing time: {total_time/60:.2f} minutes')
    print('Time breakdowns:')
    for task, time_taken in execution_times.items():
        print(f'{task}: {time_taken/60:.2f} minutes')
    
def main():
    parser = argparse.ArgumentParser(description="Process images.")
    parser.add_argument("--input", "-i", type=str, help="filepath of the input image")
    parser.add_argument("--ref", "-r", type=str, help="filepath of the reference DTM")
    parser.add_argument("--output", "-o", type=str, help="filepath of the output DTM")
    parser.add_argument("--asp", "-a", type=str, default="~/Downloads/ASP/bin", help="path to the ASP bin folder")
    parser.add_argument("--model", "-m", type=str, default="D", help="D for DenseNet-161-U-Net; V for ViT-AB-U-Net; N for NW-FC-CRF")
    parser.add_argument("--weights", "-w", type=str, default="pre-trained-weights/moon-D.pth", help="path to pretrained model")
    parser.add_argument("--tmp", "-t", type=str, default="data_tmp", help="path to a temporary working directory that can be used to store metadata")

    args = parser.parse_args()
    full_chain(args.input, args.ref, args.output, args.asp, args.model, args.weights, args.tmp)

if __name__ == "__main__":
    main()
