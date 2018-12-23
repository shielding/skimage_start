#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
"""

from skimage import data, io, color,img_as_ubyte
import pywt
import matplotlib.pyplot as plt
import numpy as np


img = data.astronaut()
img_gray = color.rgb2gray(img)
# print(img_gray.shape)
watermark = io.imread('好好学习-置乱.jpg')
# io.imshow(watermark)

for y in range(64):       
    for x in range(64):   
        # if img[y][x][0]>200 or img[y][x][1]>200 or img[y][x][2]>200:
        #     img[y][x][:3]=255
        # else:
        #     img[y][x][:3]=0
        if watermark[y][x]>160:
            watermark[y][x]=255 # 白
        else:
            watermark[y][x]=0 # 黑

# print(watermark)


# 存储M*M个（此例为64*64个低频近似子带的2*2系数）
block_cA1 = np.zeros([64,64,4,4])
block_cA2 = np.zeros([64,64,2,2]) 
block_cA3 = np.zeros([64,64]) 
img_rec = np.zeros([512,512]) # reconstructed img
mark = np.zeros([64,64]) 

def jnd(y,x):
    s = 4 * block_cA3[y][x]
    lum = 3 + 1/256 * s
    tex = 16*16*(block_cA3[y][x])
    f = 0.32
    ret = f * (lum + tex)
    # print(ret)
    return ret

jnd_value = np.zeros([64,64])

# maxm = 0 记录重构图片的像素中的最大值
num1 = 0

for y in range(64):       
    for x in range(64): 
        block = np.zeros([8,8])
        for i in range(8):
            block[i] = img_gray[y*8+i][8*x:8*(x+1)]
        coeffs1 = pywt.dwt2(block, 'haar') # 进行两次小波变换
        cA, (cH, cV, cD) = coeffs1 
        block_cA1[y][x] = cA # 4*4
        coeffs2 = pywt.dwt2(cA, 'haar')
        cA, (cH, cV, cD) = coeffs2
        block_cA2[y][x] = cA # 2*2
        new_cA2 = cA
        coeffs3 = pywt.dwt2(cA, 'haar')
        cA, (cH, cV, cD) = coeffs3
        block_cA3[y][x] = cA # 1*1
        
        jnd_value[y][x] = jnd(y,x)
        
        # new_cA = np.zeros([4,4])
        
        if np.fabs(new_cA2[0][0]-new_cA2[0][1])>np.fabs(new_cA2[1][0]-new_cA2[1][1]):
            mark[y][x] = 1 # 标记
            if watermark[y][x] == 255:
                # if new_cA2[1][1] <= new_cA2[1][0]:
                new_cA2[1][0] -= jnd_value[y][x]
                new_cA2[1][1] += jnd_value[y][x]
                if new_cA2[1][1] > new_cA2[1][0]:
                    num1+=1
            else:
                # if new_cA2[1][1] >= new_cA2[1][0]:
                new_cA2[1][0] += jnd_value[y][x]
                new_cA2[1][1] -= jnd_value[y][x]
                if new_cA2[1][1] < new_cA2[1][0]:
                    num1+=1
        else:
            if watermark[y][x] == 255:
                # if new_cA2[0][1] <= new_cA2[0][0]:
                new_cA2[0][0] -= jnd_value[y][x]
                new_cA2[0][1] += jnd_value[y][x]
                if new_cA2[0][1] > new_cA2[0][0]:
                    num1+=1
            else:
                # if new_cA2[0][1] >= new_cA2[0][0]:
                new_cA2[0][0] += jnd_value[y][x]
                new_cA2[0][1] -= jnd_value[y][x]
                if new_cA2[0][1] < new_cA2[0][0]:
                    num1+=1
        # 将更改过的低频系数赋给coeffs2
        coeffs2 = list(coeffs2)
        coeffs2[0] = new_cA2
        coeffs2 = tuple(coeffs2)
        # 根据小波变换的系数重构图片
        new_cA1 = pywt.idwt2(coeffs2, 'haar')
        coeffs1 = list(coeffs1)
        coeffs1[0] = new_cA1
        coeffs1 = tuple(coeffs1)
        block_rec = pywt.idwt2(coeffs1, 'haar')
        for i in range(8):
            np.clip(block_rec[i],0,1,out = block_rec[i]) # 确保浮点数在0到1之间
            img_rec[y*8+i][8*x:8*(x+1)] = block_rec[i] 
            # for k in block_rec[i]:
            #     if k > maxm:
            #         maxm = k
  
# 调试代码      
# print(jnd_value)     
# print(cA.shape)
# print(block_cA1.shape)
# print(block_cA2.shape)
# print(block_cA3.shape)
# print(cA)
# io.imshow(img_gray)
# plt.imshow(img_gray,plt.cm.gray) 
# plt.imshow(img_rec,plt.cm.gray)
# print(img_rec) 
# print(img_gray)
# print(maxm)

# io.imshow(img_rec)


# cnt = 0
# for i in range(512):
#     for j in range(512):
#         if img_rec[i][j] != img_gray[i][j]:
#             cnt += 1
# print(cnt) # 23万
im_uint8 = img_as_ubyte(img_rec)
io.imsave("嵌入水印的图像.jpg",im_uint8)
io.imshow(img_rec)
# print(im_uint8)
print(num1) #4000上下
img_embedded = io.imread('嵌入水印的图像.jpg')
# img_embedded = img_rec
watermark = np.zeros([64,64]) 

for y in range(64):       
    for x in range(64): 
        block = np.zeros([8,8])
        for i in range(8):
            block[i] = img_embedded[y*8+i][8*x:8*(x+1)]
        coeffs1 = pywt.dwt2(block, 'haar') # 进行两次小波变换
        cA, (cH, cV, cD) = coeffs1 
        coeffs2 = pywt.dwt2(cA, 'haar')
        cA, (cH, cV, cD) = coeffs2
        if mark[y][x]==1:
            if cA[1][1]>cA[1][0]:
                watermark[y][x] = 255
        else:
            if cA[0][1]>cA[0][0]:
                watermark[y][x] = 255
  
# print(watermark)    
          
w_uint8 = np.array(watermark,dtype=np.uint8)
# io.imshow(w_uint8)
io.imsave("提取的水印.jpg",w_uint8)
            




