#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sobel算子可以用来检测边缘
@author: shielding
"""

from skimage import data, filters,io,color

img = color.rgb2gray(data.coffee())

edges = filters.sobel(img)

io.imshow(edges)

io.imsave("edge.png",edges)