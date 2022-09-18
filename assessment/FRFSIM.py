import numpy as np
import cv2

BINS = 256
MAX_LEVEL = BINS - 1

if __name__ == "__main__":
    D = cv2.imread('assessment/data/GRCN.png')
    R = cv2.imread('assessment/data/sbte.jpg')

    Dg = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY) / MAX_LEVEL
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY) / MAX_LEVEL

    ho, wo = D.shape
    hc, wc = R.shape
    ratio_orig = ho/wo
    ratio_comp = hc/wc
    dim = (wc, hc)