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

def get_value(img):
    return np.average(img, 2)


def transform(img, I): # img: depth, I: Hazy image
    h, w = img.shape[:2]
    near_image = np.zeros(img.shape, img.dtype)
    far_image = np.zeros(img.shape, img.dtype)
    for i in range(h):
        for j in range(w):
            if(img[i][j] >= 110): 
                near_image[i][j] = img[i][j] 
            else:
                far_image[i][j] = img[i][j] 
    # 각각 I랑 합성시키고, near_image에서는 I의 배경이 들어가고
    # far_image에는 I의 사람이 들어감
    # I랑 합성한 near_image를 SBTE 알고리즘 돌려
    # 그러면 지금 나온 게 far_image의 정상인 사람과 안개제거된 배경이 있음
    # 나온 결과 두 개를 합성함.             
    cv2.imshow('near', near_image)
    cv2.imshow('far', far_image)
    cv2.imwrite('dehaze/outputs/near_image.jpg', near_image)
    cv2.imwrite('dehaze/outputs/far_image.jpg', far_image)
    return near_image, far_image


if __name__ == "__main__":
    # DCP 방법처럼 갈게 일단은 -> 우선 rgb를 split하고, 걔네들의 명도를 각각 따로 구하고 transform 처리한 후 return 할 때 다 더해주면 됨
    I = cv2.imread('data/haze_image.jpg')
    depth_I = cv2.imread('data/haze_depth.jpg')
    v = get_value(depth_I)
    a, b = transform(v, I)
    # result1 = cv2.merge([h, s, a])
    # result2 = cv2.merge([h, s, b])
    # cv2.imshow('test', v)
    # cv2.imshow('rseult', result1)
    # cv2.imshow('rseult', result2)
    # cv2.imwrite('dehaze/outputs/value of depth.jpg', v)
    cv2.waitKey()
