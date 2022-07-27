import cv2
from cv2 import imwrite
import numpy as np
from os import path as ospath

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

    masked = cv2.copyTo(I, near_image, sbte)
    cv2.imshow('masked', masked)

    return masked


if __name__ == "__main__":
    DIR = 'dehaze/outputs'
    FILE_NAME = 'forest.jpg'
    O_PATH = ospath.join(DIR, FILE_NAME)
    I_PATH = ospath.join('./data/hazy', FILE_NAME)
    DEPTH_I_PATH = ospath.join('./data/depth', FILE_NAME)
    SBTE_PATH = ospath.join('dehaze/sbte_result', FILE_NAME)

    I = cv2.imread(I_PATH)
    depth_I = cv2.imread(DEPTH_I_PATH)
    sbte = cv2.imread(SBTE_PATH)
    v = cv2.cvtColor(depth_I, cv2.COLOR_BGR2GRAY)
    result = transform(v, I, sbte)
    cv2.imwrite(O_PATH, result)

    cv2.waitKey()
