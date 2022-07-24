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
    near_image = np.zeros(img.shape, img.dtype)
    far_image = np.zeros(img.shape, img.dtype)
    for i in range(h):
        for j in range(w):
            if(img[i][j] > BINS / 2):  # 우선 명도의 절반만큼
                near_image[i][j] = img[i][j]
            else:
                far_image[i][j] = img[i][j]
    cv2.imshow('near', near_image)
    cv2.imshow('far', far_image)
    cv2.imwrite('dehaze/outputs/near image.jpg', near_image)
    cv2.imwrite('dehaze/outputs/far image.jpg', far_image)
    return near_image, far_image


if __name__ == "__main__":
    # DCP 방법처럼 갈게 일단은 -> 우선 rgb를 split하고, 걔네들의 명도를 각각 따로 구하고 transform 처리한 후 return 할 때 다 더해주면 됨
    depth_I = cv2.imread('data/haze_depth.jpg')
    # hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # cvtColor(I, cv2.COLOR_BGR2GRAY)
    for i in range(3):


    near_result, far_result = transform(v)
    # result = cv2.merge([h, s, near_result])
    # result = cv2.merge([h, s, far_result])
    cv2.imshow('test', v)
    cv2.imshow('rseult', result)
    cv2.imwrite('dehaze/outputs/value of depth.jpg', v)
    cv2.waitKey(0)
