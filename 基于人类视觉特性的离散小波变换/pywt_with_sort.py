#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
"""

from skimage import data, io, color,img_as_ubyte
import pywt
# import matplotlib.pyplot as plt
import numpy as np
import math
from operator import attrgetter

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
cH1 = np.zeros([64,64,4,4])  
cD1 = np.zeros([64,64,4,4])  
cV1 = np.zeros([64,64,4,4])  
cH2 = np.zeros([64,64,2,2])
cV2 = np.zeros([64,64,2,2])
cD2 = np.zeros([64,64,2,2])
block_cA3 = np.zeros([64,64]) 
img_rec = np.zeros([512,512]) # reconstructed img
mark = np.zeros([64,64]) 

def jnd(y,x):
    s = 4 * block_cA3[y][x]
    lum = 3 + 1/256 * s
    tex = 1/16*1/16*(block_cA3[y][x])
    f = 0.04
    ret = f * (lum + tex)
    # print(ret)
    return ret

jnd_value = np.zeros([64,64])

# maxm = 0 记录重构图片的像素中的最大值
num1 = 0

class one:
    def __init__(self,y,x,c):
        self.y = y
        self.x = x
        self.c = c
    # def get_c(self):
    #     return self.c
    def show(self):
        print(self.y,'-',self.x,'-',self.c)

ones = []

for y in range(64):       
    for x in range(64): 
        block = np.zeros([8,8])
        for i in range(8):
            block[i] = img_gray[y*8+i][8*x:8*(x+1)]
        coeffs1 = pywt.dwt2(block, 'haar') # 一级小波变换
        cA, (cH, cV, cD) = coeffs1 
        block_cA1[y][x] = cA # 4*4
        cH1[y][x] = cH
        cV1[y][x] = cV
        cD1[y][x] = cD
        coeffs2 = pywt.dwt2(cA, 'haar') # 二级小波变换
        cA, (cH, cV, cD) = coeffs2
        block_cA2[y][x] = cA # 2*2
        cH2[y][x] = cH
        cV2[y][x] = cV
        cD2[y][x] = cD
        old_cA2 = cA
        coeffs3 = pywt.dwt2(cA, 'haar') # 三级小波变换
        cA, (cH, cV, cD) = coeffs3
        block_cA3[y][x] = cA # 1*1
        
        jnd_value[y][x] = jnd(y,x)
        
        ones.append(one(y*2,x*2,old_cA2[0][0]))
        ones.append(one(y*2,x*2+1,old_cA2[0][1]))
        ones.append(one(y*2+1,x*2,old_cA2[1][0]))
        ones.append(one(y*2+1,x*2+1,old_cA2[1][1]))
    
# print(len(ones))

    
ones.sort(key=attrgetter('c'))
# for i in range(12800,12890):
#     ones[i].show()      

ones_map = np.zeros([128,128,2])

for i in range(128):
    for j in range(128):
        ones_map[i][j][0] = ones[i*128+j].y
        ones_map[i][j][1] = ones[i*128+j].x

def find(o):
    return ones_map[o.y][o.x]

for i in range(64):
    for j in range(64):  
        new_cA2 = np.zeros([2,2])
        new_cA2[0][0] = ones[i*64*4+j*2].c
        x1 = math.floor(ones[i*64*4+j*2].x/2)
        y1 = math.floor(ones[i*64*4+j*2].y/2)
        new_cA2[0][1] = ones[i*64*4+j*2+1].c
        x2 = math.floor(ones[i*64*4+j*2+1].x/2)
        y2 = math.floor(ones[i*64*4+j*2+1].y/2)
        new_cA2[1][0] = ones[i*64*4+128+j*2].c
        x3 = math.floor(ones[i*64*4+128+j*2].x/2)
        y3 = math.floor(ones[i*64*4+128+j*2].y/2)
        new_cA2[1][1] = ones[i*64*4+128+j*2+1].c
        x4 = math.floor(ones[i*64*4+128+j*2+1].x/2)
        y4 = math.floor(ones[i*64*4+128+j*2+1].y/2)
         
        if np.fabs(new_cA2[0][0]-new_cA2[0][1])>np.fabs(new_cA2[1][0]-new_cA2[1][1]):
            mark[i][j] = 1 # 标记
            if watermark[i][j] == 255:
                if new_cA2[1][1] <= new_cA2[1][0]:
                    new_cA2[1][0] -= jnd_value[y3][x3]
                    new_cA2[1][1] += jnd_value[y4][x4]
                if new_cA2[1][1] > new_cA2[1][0]:
                    num1+=1
            else:
                if new_cA2[1][1] >= new_cA2[1][0]:
                    new_cA2[1][0] += jnd_value[y3][x3]
                    new_cA2[1][1] -= jnd_value[y4][x4]
                if new_cA2[1][1] < new_cA2[1][0]:
                    num1+=1
        else:
            if watermark[i][j] == 255:
                if new_cA2[0][1] <= new_cA2[0][0]:
                    new_cA2[0][0] -= jnd_value[y1][x1]
                    new_cA2[0][1] += jnd_value[y2][x2]
                if new_cA2[0][1] > new_cA2[0][0]:
                    num1+=1
            else:
                if new_cA2[0][1] >= new_cA2[0][0]:
                    new_cA2[0][0] += jnd_value[y1][x1]
                    new_cA2[0][1] -= jnd_value[y2][x2]
                if new_cA2[0][1] < new_cA2[0][0]:
                    num1+=1
                    
        ones[i*64*4+j*2].c = new_cA2[0][0]
        ones[i*64*4+j*2+1].c = new_cA2[0][1] 
        ones[i*64*4+128+j*2].c = new_cA2[1][0] 
        ones[i*64*4+128+j*2+1].c = new_cA2[1][1] 
        
        # if i==30:
        #     print(ones[i*64*4+j*2].c,'-',ones[i*64*4+j*2+1].c,'-',ones[i*64*4+128+j*2].c,'-',ones[i*64*4+128+j*2+1].c)
                    
print(num1)
ones.sort(key=attrgetter('y','x'))
# for i in range(400,600):
#     ones[i].show()
#     print(i)

    
for y in range(64):       
    for x in range(64): 
        new_cA2 =  np.zeros([2,2])
        new_cA2[0][0] = ones[y*64*4+x*2].c
        new_cA2[0][1] = ones[y*64*4+x*2+1].c
        new_cA2[1][0] = ones[y*64*4+128+x*2].c
        new_cA2[1][1] = ones[y*64*4+128+x*2+1].c
        # 将更改过的低频系数赋给coeffs2
        coeffs2 = new_cA2,(cH2[y][x],cV2[y][x],cD2[y][x])
        # print(coeffs2)
        coeffs2 = tuple(coeffs2)
        # 根据小波变换的系数重构图片
        new_cA1 = pywt.idwt2(coeffs2, 'haar')
        coeffs1 = new_cA1,(cH1[y][x],cV1[y][x],cD1[y][x])
        coeffs1 = tuple(coeffs1)
        block_rec = pywt.idwt2(coeffs1, 'haar')
        for i in range(8):
            np.clip(block_rec[i],0,1,out = block_rec[i]) # 确保浮点数在0到1之间
            img_rec[y*8+i][8*x:8*(x+1)] = block_rec[i] 
        
        
        
# im_uint8 = img_as_ubyte(img_rec)
# io.imsave("嵌入水印的图像.jpg",im_uint8)
io.imsave("嵌入水印的图像.jpg",img_rec)
# io.imshow(img_rec)

# 提取

# img_embedded = io.imread('嵌入水印的图像.jpg') 


img_gaussian4 = util.random_noise(img_rec, mode='gaussian',var = 0.004)

img_gaussian6 = util.random_noise(img_rec, mode='gaussian',var = 0.006)

img_gaussian8 = util.random_noise(img_rec, mode='gaussian',var = 0.008)

img_gaussian1 = util.random_noise(img_rec, mode='gaussian',var = 0.01)

img_sp1 = util.random_noise(img_rec, mode='s&p',amount = 0.01)

img_sp2 = util.random_noise(img_rec, mode='s&p',amount = 0.02)

img_sp3 = util.random_noise(img_rec, mode='s&p',amount = 0.03)

img_sp5 = util.random_noise(img_rec, mode='s&p',amount = 0.05)

img_embedded = img_gaussian1
watermark = np.zeros([64,64]) 


new_ones = []

for y in range(64):       
    for x in range(64): 
        block = np.zeros([8,8])
        for i in range(8):
            block[i] = img_embedded[y*8+i][8*x:8*(x+1)]
        coeffs1 = pywt.dwt2(block, 'haar') # 一级小波变换
        cA, (cH, cV, cD) = coeffs1 
        block_cA1[y][x] = cA # 4*4
        cH1[y][x] = cH
        cV1[y][x] = cV
        cD1[y][x] = cD
        coeffs2 = pywt.dwt2(cA, 'haar') # 二级小波变换
        cA, (cH, cV, cD) = coeffs2
        block_cA2[y][x] = cA # 2*2
        cH2[y][x] = cH
        cV2[y][x] = cV
        cD2[y][x] = cD
        old_cA2 = cA
        coeffs3 = pywt.dwt2(cA, 'haar') # 三级小波变换
        cA, (cH, cV, cD) = coeffs3
        block_cA3[y][x] = cA # 1*1
        
        jnd_value[y][x] = jnd(y,x)
        
        new_ones.append(one(y*2,x*2,old_cA2[0][0]))
        new_ones.append(one(y*2,x*2+1,old_cA2[0][1]))
        new_ones.append(one(y*2+1,x*2,old_cA2[1][0]))
        new_ones.append(one(y*2+1,x*2+1,old_cA2[1][1]))
    
new_ones.sort(key=attrgetter('y','x'))

map_c = np.zeros([128,128])
for i in new_ones:
    map_c[i.y][i.x] = i.c
 
new_map = np.zeros([128,128])
    
for i in range(128):
    for j in range(128): 
        y = int(ones_map[i][j][0])
        x = int(ones_map[i][j][1])
        new_map[i][j] = map_c[y][x]
    
# for i in range(0,90):
#     print(ones_map[100][i][0], ones_map[100][i][1],new_map[100][i])  
# print(len(new_ones))
# new_ones.sort(key = find)
# for i in range(400,600):
#     new_ones[i].show()
#     print(i)
   
# print(watermark)   



for i in range(64):
    for j in range(64):  
        new_cA2 = np.zeros([2,2])
        new_cA2[0][0] = new_map[i*2][j*2]
        new_cA2[0][1] = new_map[i*2][j*2+1]
        new_cA2[1][0] = new_map[i*2+1][j*2]
        new_cA2[1][1] = new_map[i*2+1][j*2+1]
        # if i==30:
        #     print(new_cA2[0][0],'-',new_cA2[0][1],'-',new_cA2[1][0],'-',new_cA2[1][1])
        if mark[i][j] == 1:
            if new_cA2[1][1]>new_cA2[1][0]:
                watermark[i][j] = 255
                # if i==30:
                #     print('255')
        else:
            if new_cA2[0][1]>new_cA2[0][0]:
                watermark[i][j] = 255
                # if i==30:
                #     print('255')
                

w_uint8 = np.array(watermark,dtype=np.uint8)
io.imshow(w_uint8)
io.imsave("提取的水印.jpg",w_uint8)


