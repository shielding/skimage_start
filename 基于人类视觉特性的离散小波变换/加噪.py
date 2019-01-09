#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
"""

from skimage import io, util

img_embedded = io.imread('嵌入水印的图像.jpg')
img_gaussian = util.random_noise(img_embedded, mode='gaussian',var = 0.004)
io.imsave("0.004高斯.jpg",img_gaussian)
img_gaussian = util.random_noise(img_embedded, mode='gaussian',var = 0.006)
io.imsave("0.006高斯.jpg",img_gaussian)
img_gaussian = util.random_noise(img_embedded, mode='gaussian',var = 0.008)
io.imsave("0.008高斯.jpg",img_gaussian)
img_sp = util.random_noise(img_embedded, mode='s&p',amount = 0.01)
io.imsave("0.01椒盐.jpg",img_sp)
img_sp = util.random_noise(img_embedded, mode='s&p',amount = 0.02)
io.imsave("0.02椒盐.jpg",img_sp)
img_sp = util.random_noise(img_embedded, mode='s&p',amount = 0.03)
io.imsave("0.03椒盐.jpg",img_sp)