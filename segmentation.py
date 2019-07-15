#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:08:08 2019

@author: ashwathcs
"""
#import dependencies
import cv2
import numpy as np

class convert:    
    def morphological_transformation(image_path):
        #read image using opencv
        img = cv2.imread(image_path,1)
        #convert image to RBG
        segmented = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #smoothning the image
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(img,-1,kernel)
        #converting image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:,:,1]
        #median blur of image for removing the salt and pepper noise 
        median = cv2.medianBlur(sat,5)
        kernel1 = np.ones((5,5),np.float32)/25
        kernel2 = np.array([[-1,-1,-1], [-1,11,-1], [-1,-1,-1]]) 
        #th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        blur = cv2.GaussianBlur(median,(5,5),0)
        #applying OTSU thresholding
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        dst = cv2.filter2D(th3,-1, kernel1)
        im = cv2.filter2D(dst, -1, kernel2)
        #applying the binary mask to remove unwanted regions of image
        binary_mask = np.array(im,dtype='bool') 
        binary = np.array(binary_mask,dtype = 'uint8')
        #dilation on the binary image , adds pixels to the boundary
        kernel = np.ones((5,5),np.uint8)
        binary_mask= cv2.dilate(binary,kernel,iterations = 1)
        segmented[:,:,0] = np.multiply(segmented[:,:,0],binary_mask)
        segmented[:,:,1] = np.multiply(segmented[:,:,1],binary_mask)
        segmented[:,:,2] = np.multiply(segmented[:,:,2],binary_mask)
        return segmented,binary_mask

