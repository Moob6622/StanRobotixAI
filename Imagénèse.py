# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:41:47 2018

@author: Master
"""

import numpy
import matplotlib.pyplot as plt
import cv2
import time

class Imagenese (object) :

    def __init__(self, image) :
        self.image = cv2.imread(image+'.jpg', 0)
        
    def Rotate (self) :
        rows, cols = self.image.shape

        
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst = cv2.warpAffine(self.image, M, (cols,rows))
        cv2.imshow('Rotated Image', self.image)        
    
import Imagénèse
x = Imagénèse.Imagenese('cube1')
x.Rotate()
        
        

                
    

