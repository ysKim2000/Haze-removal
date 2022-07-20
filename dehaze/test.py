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

if __name__ == "__main__":
    I = cv2.imread('dehaze/haze_depth.jpg')
    print(I)
    print(type(I))
    cv2.imshow('test',I)
    cv2.waitKey()
    