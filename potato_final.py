#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:20:07 2019

@author: ashwathcs
"""
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from segmentation import convert
from getborders import borders
from predict_potato import predict
import cv2
img = cv2.imread('late3.JPG')
segmented,binary_mask = convert.morphological_transformation('late3.JPG')
edges = borders.get_borders(binary_mask)
disease_detected = predict.predict_disease(segmented)
img_list = []
img_list.append(img)
img_list.append(binary_mask)
img_list.append(edges)
img_list.append(segmented)
print(disease_detected)
for i in img_list:
    plt.figure()
    plt.imshow(i)

subplot(2,2,1)
plt.imshow(img_list[0])
title('subplot(2,2,1)')
 
subplot(2,2,2)
plt.imshow(img_list[1])
title('subplot(2,2,2)')

 
subplot(2,2,3)
plt.imshow(img_list[2])
title('subplot(2,2,3)')
 
subplot(2,2,4)
plt.imshow(img_list[3])
title('subplot(2,2,4)')
 
show()

