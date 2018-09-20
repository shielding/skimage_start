#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skimage加随机噪声的函数为：
skimage.util.random_noise(image, mode=’gaussian’, seed=None,mean=0, clip=True, **kwargs)
@author: shielding
"""

from skimage import io, data, color, util

# src = io.imread('    .png') # 读入格式为numpy数组

coffee = data.coffee()
img = coffee

img_gray = color.rgb2gray(img) # 转为灰度图像，还可以使用convert_colorspace

img_gaussian = util.random_noise(img_gray, mode='gaussian', clip=True)
img_sp = util.random_noise(img_gray, mode='s&p')

io.imshow(img_gaussian)
io.imshow(img_sp)

io.imsave("img_gaussian.png",img_gaussian)
io.imsave("img_sp.png",img_sp)


# dst = io.imshow(img) # 类型为'matplotlib.image.AxesImage',和plt的绘图一致
# io.imshow(img_gray)
# dst = plt.imshow(img)# 可选择颜色图谱，默认RGB
# print(type(dst))

# print(img.shape) # 高度，宽度，通道数
# print(img_gray.shape)


