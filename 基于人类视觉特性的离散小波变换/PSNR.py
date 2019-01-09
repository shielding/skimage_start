#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
"""

from skimage import data, io, color, util
import numpy as np
import math

img = data.astronaut()
img_gray = color.rgb2gray(img)
img_int = util.img_as_ubyte(img_gray)
img_embedded = io.imread('嵌入水印的图像.jpg')

print(img_int)
print(img_embedded)
 
def psnr(img1: np.ndarray, img2: np.ndarray):
    mse = np.mean((img1 - img2) ** 2) 
    return 20 * np.log10(255 / mse**0.5)

print(psnr(img_int,img_embedded))

img1 = io.imread('好好学习-二值化.jpg')
img2 = io.imread('好好学习-反置乱.jpg')

def nc():
    d = 0
    f1 = 0
    f2 = 0
    for i in range(64):
        for j in range(64):
            d += img1[i][j]*img2[i][j]
            f1 += img1[i][j]*img1[i][j]
            f2 += img2[i][j]*img2[i][j]
    return d/(math.sqrt(f1))/(math.sqrt(f2))

print(nc())        
    