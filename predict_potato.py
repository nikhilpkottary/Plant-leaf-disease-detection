#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 08:52:09 2019

@author: ashwathcs
"""
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model
class predict:
    def predict_disease(segmented):
        
        classifier = load_model('potato_disease1.h5')
        
        labels = {0:'Potato Early blight',1:'Potato late blight',2:'Potato healthy leaf'}
        #converting image to numpy array for processing
        segmented = cv2.resize(segmented,(64,64))
        test_image = image.img_to_array(segmented)
        
        test_image = np.expand_dims(test_image, axis = 0)
        #image normalisation to reduce the range of pixel values
        test_image = test_image/255
        
        result = classifier.predict(test_image)
        disease_detected = labels[np.argmax(result[0],axis=0)]
        return disease_detected
