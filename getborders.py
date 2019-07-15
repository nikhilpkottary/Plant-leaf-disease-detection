#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:20:52 2019

@author: ashwathcs
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
#convert image to RBG
class borders:
    def get_borders(binary):
        binary = np.array(binary,dtype = 'uint8')
        #dilation on the binary image , adds pixels to the boundary
        kernel = np.ones((5,5),np.uint8)
        binary= cv2.dilate(binary,kernel,iterations = 1)
        edgeX = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
        edgeY = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
        edgeX = np.uint8(np.absolute(edgeX))
        edgeY = np.uint8(np.absolute(edgeY))
        edge = cv2.bitwise_or(edgeX, edgeY)
        return edge

