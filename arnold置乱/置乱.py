#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
"""


import numpy as np
from skimage import data, io, color
# import matplotlib.pyplot as plt


im_data = np.zeros([64,64,4])#所使用的图片的大小为400*400
img = io.imread('好好学习.png') #将图片读成rgb的numpy array格式
# img = data.astronaut()
# print("the picture's shape: ",img.shape)
# print(img)

a = 1
b = 1
N = 64
for i in range(1,37):   
    for y in range(64):       
        for x in range(64):   
            xx = (x+b*y)%N           
            yy = ((a*x)+(a*b+1)*y)%N  
             # print(img[y][x])
            im_data[yy][xx] = img[y][x]   
    im_uint8 = np.array(im_data,dtype=np.uint8)
    img = im_uint8


# io.imshow(im_data)
# print(img)
im_uint8 = np.array(im_data,dtype=np.uint8)
io.imsave("好好学习-置乱.png",im_uint8)
# io.imsave('astronaut.jpg',img)