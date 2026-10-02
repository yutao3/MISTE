#!/usr/bin/env python

from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error
from math import sqrt
import argparse
import imutils
import cv2
import os
import sys

def calc_stat(input_A, input_B):
    imageA = cv2.imread(input_A, 0)
    imageB = cv2.imread(input_B, 0)
    
    (score, diff) = compare_ssim(imageA, imageB, full=True)
    print("SSIM: {}".format(score))
    
    rms = sqrt(mean_squared_error(imageA, imageB))
    print("RMSE: {}".format(rms))
    
if __name__ == "__main__":
    calc_stat(sys.argv[1],sys.argv[2])
