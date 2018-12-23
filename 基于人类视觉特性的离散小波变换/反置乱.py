#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
"""

import numpy as np
from skimage import data, io
# import matplotlib.pyplot as plt


im_data = np.zeros([64,64])#所使用的图片的大小为400*400
# img = io.imread('好好学习-置乱.jpg')
img = io.imread('提取的水印.jpg')
# img = data.astronaut()
print("the picture's shape: ",img.shape)

for y in range(64):       
    for x in range(64):   
        # if img[y][x][0]>200 or img[y][x][1]>200 or img[y][x][2]>200:
        #     img[y][x][:3]=255
        # else:
        #     img[y][x][:3]=0
        if img[y][x]>160:
            img[y][x]=255 # 白
        else:
            img[y][x]=0 # 黑

print(img)

a = 1
b = 1
N = 64
for i in range(1,37):
    for y in range(64):
        for x in range(64):
            xx = ((a*b+1)*x-b*y)%N
            yy = (-a*x+y)%N
            im_data[yy][xx] = img[y][x] 
    im_uint8 = np.array(im_data,dtype=np.uint8)
    img = im_uint8
    # img = im_data

# print(img)


im_uint8 = np.array(im_data,dtype=np.uint8)
io.imshow(im_uint8)
io.imsave("好好学习-反置乱.jpg",im_uint8)
# im_float64 = np.array(im_data,dtype=np.float64)
# io.imsave("好好学习-反置乱.jpg",im_float64)