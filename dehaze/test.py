import cv2
import numpy as np
import os
from skimage import morphology
import matplotlib.pyplot as plt

BINS = 256
MAX_LEVEL = BINS - 1

def i2f(img):
    return img / MAX_LEVEL

def f2i(img):
    return np.uint8(np.around(img * MAX_LEVEL))

# def segmentation(img):
#     original = 

if __name__ == "__main__":
    I = cv2.imread('dehaze/haze_depth.jpg')
    hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    print(v)
    print(type(v))
    print(np.shape(v))
    cv2.imshow('test',v)
    cv2.waitKey()
    