#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
""" 
from skimage import exposure
from skimage import data
import matplotlib.pyplot as plt

img = data.coffee()
arr = img.flatten()

plt.figure(num = "different process")

plt.subplot(221)
plt.title("origin") 
plt.imshow(img)
plt.axis("off")

plt.subplot(222)
plt.title("histogram") 
plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='blue')  

hist = exposure.equalize_hist(img, nbins=256, mask=None)
plt.subplot(223)
plt.title("equalization")
plt.imshow(hist,plt.cm.gray)
plt.axis('off') 

plt.subplot(224)
plt.title("sigmoid")
plt.imshow(exposure.adjust_sigmoid(img, cutoff=0.5, gain=10, inv=False))
plt.axis('off')

plt.tight_layout() 
plt.show()

