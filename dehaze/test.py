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


def transform(img):
    h, w = img.shape[:2]
    new_image = np.zeros(img.shape, img.dtype)
    for i in range(h):
        for j in range(w):
            if(img[i][j] > 125):  # 우선 명도의 절반만큼
                new_image[i][j] = img[i][j]
    cv2.imshow('test2', new_image)
    cv2.imwrite('dehaze/outputs/reuslt.jpg', new_image)
    return new_image


if __name__ == "__main__":
    # DCP 방법처럼 갈게 일단은 -> 우선 rgb를 split하고, 걔네들의 명도를 각각 따로 구하고 transform 처리한 후 return 할 때 다 더해주면 됨
    I = cv2.imread('data/haze_depth.jpg')
    hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    test_result = transform(v)
    result = cv2.merge([h, s, test_result])
    cv2.imshow('test', v)
    cv2.imshow('rseult', result)
    cv2.imwrite('dehaze/outputs/value of depth.jpg', v)
    cv2.waitKey()
