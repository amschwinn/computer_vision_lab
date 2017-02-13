# -*- coding: utf-8 -*-
"""
Computer Vision Lab 1
MLDM University Jean Monnet
Austin Schwinn
Feb 13, 2017

Implementing Harris Corner Detection.
"""
__author__ = "amschwinn"

import os
from skimage import io
import numpy as np
import scipy as sp
from scipy import signal
from PIL import Image
'''
PART 1
Read image, compute image derivative lx and ly,
and apply the smoothing filter
'''
#Set wd
os.chdir('C:/Users/amsch/OneDrive/Documents/OneDrive/Documents/MLDM/Computer Vision/Lab Session 1_Materials')
#load file
chessboard = io.imread('chessboard00.png')
#Compute the image derivative lx and  ly
#compute image gradient for each pixel in x and y directions
dy, dx = np.gradient(chessboard)


#Generate Gaussian Filter of size 9*9 and stand dev 2
def gauss2d(shape=(9,9), sigma=2):
    m,n = [(ss-1)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1, - n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h [ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

gauss_filter = gauss2d()

#compute dx^2 dy^2 and dxdy
Ixx = dx**2
Iyy = dy**2
Ixy = dy*dx

#apply gaussian filter to Ixx, Iyy, Ixy
im_xx = signal.convolve(Ixx,gauss_filter,mode='same')
im_yy = signal.convolve(Iyy,gauss_filter,mode='same')
im_xy = signal.convolve(Ixy,gauss_filter,mode='same')

#Display the results
print im_xx
print im_yy
print im_xy

'''
Part 2
Compute E, the matrix that contains for each pixel the value
of the smaller eigenvalue of M. Display the matrix E  
'''

