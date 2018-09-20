#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用matplotlib绘图
@author: shielding
"""

from skimage import data
import matplotlib.pyplot as plt

coffee = data.coffee()
img = coffee

plt.figure(num='coffee',figsize=(8,8)) 

plt.subplot(2,2,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
plt.title('origin image')   #第一幅图片标题
plt.imshow(img)      #绘制第一幅图片

plt.subplot(2,2,2)     #第二个子图
plt.title('R channel')   #第二幅图片标题
plt.imshow(img[:,:,0],plt.cm.gray)      #绘制第二幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

plt.subplot(2,2,3)     #第三个子图
plt.title('G channel')   #第三幅图片标题
plt.imshow(img[:,:,1],plt.cm.gray)      #绘制第三幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

plt.subplot(2,2,4)     #第四个子图
plt.title('B channel')   #第四幅图片标题
plt.imshow(img[:,:,2],plt.cm.gray)      #绘制第四幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

plt.tight_layout()  #自动调整subplot间的参数

plt.show()   #显示窗口

# 还可以使用skimage.viewer的ImageViewer绘图