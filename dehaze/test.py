import cv2
from cv2 import imwrite
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


def transform(img, I, sbte): # img: depth, I: Hazy image, sbte: dehazy image
    h, w = img.shape[:2]
    near_image = np.zeros(img.shape, img.dtype)
    far_image = np.zeros(img.shape, img.dtype)
    for i in range(h):
        for j in range(w):
            if(img[i][j] >= 85): 
                near_image[i][j] = img[i][j] 
            else:
                far_image[i][j] = img[i][j] 
    # 각각 I랑 합성시키고, near_image에서는 I의 배경이 들어가고
    # far_image에는 I의 사람이 들어감
    # I랑 합성한 near_image를 SBTE 알고리즘 돌려
    # 그러면 지금 나온 게 far_image의 정상인 사람과 안개제거된 배경이 있음
    # 나온 결과 두 개를 합성함.     
    print(np.shape(I))
    print(np.shape(near_image))
    print(np.shape(sbte))
    masked = cv2.copyTo(I, near_image, sbte)
    cv2.imshow('masked', masked)
    cv2.imwrite('dehaze/outputs/result.jpg', masked)
    
    cv2.imshow('near', near_image)
    cv2.imshow('far', far_image)
    cv2.imwrite('dehaze/outputs/near_image.jpg', near_image)
    cv2.imwrite('dehaze/outputs/far_image.jpg', far_image)
    return near_image, far_image


if __name__ == "__main__":
    I = cv2.imread('data/haze_image.jpg')
    depth_I = cv2.imread('data/haze_depth.jpg')
    sbte = cv2.imread('data/sbte.jpg')
    # v = get_value(depth_I)
    v = cv2.cvtColor(depth_I, cv2.COLOR_BGR2GRAY)
    a, b = transform(v, I, sbte)

    cv2.waitKey()
