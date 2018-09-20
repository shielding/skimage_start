#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中值滤波 disk用于设置滤波器的形状 越大图像越模糊
@author: shielding
"""

from skimage import filters
from skimage.morphology import disk
import matplotlib.pyplot as plt

img_gaussian = plt.imread("img_gaussian.png")
img_sp = plt.imread("img_sp.png")

f1 = filters.median(img_gaussian, disk(5))
f2 = filters.median(img_sp, disk(5))
f3 = filters.median(img_gaussian, disk(8))
f4 = filters.median(img_sp, disk(8))

plt.figure('median',figsize = (8,8))

plt.subplot(221)
plt.title("img_gaussian disk:5")
plt.imshow(f1,plt.cm.gray)
plt.axis('off') 

plt.subplot(222)
plt.title("img_sp disk:5")
plt.imshow(f2,plt.cm.gray)
plt.axis('off') 

plt.subplot(223)
plt.title("img_gaussian disk:8")
plt.imshow(f3,plt.cm.gray)
plt.axis('off') 

plt.subplot(224)
plt.title("img_sp disk:8")
plt.imshow(f4,plt.cm.gray)
plt.axis('off') 

plt.tight_layout()
plt.show()
