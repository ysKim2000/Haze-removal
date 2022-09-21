import numpy as np
import math
import cv2

BINS = 256
MAX_LEVEL = BINS - 1

if __name__ == "__main__":
    D = cv2.imread('assessment/data/sbte.jpg') # Defogged image
    R = cv2.imread('assessment/data/GRCN.png') # Reference image

    # Grayscale
    Dg = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY) / MAX_LEVEL # Defogged gray image 
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY) / MAX_LEVEL # Reference gray image

    # RGB to HSV
    h_d, s_d, v_d= cv2.cvtColor(D, cv2.COLOR_BGR2HSV)
    h_r, s_r, v_r= cv2.cvtColor(R, cv2.COLOR_BGR2HSV)

    # RGB and gray channel
    D_R = D[:,:,3] # Defogged image
    D_G = D[:,:,2]
    D_B = D[:,:,1]

    R_R = R[:,:,3] # Reference image
    R_G = R[:,:,2]
    R_B = R[:,:,1]

    # MSCN
    MSCN_window = cv2.getGaussianKernel(7, 7/6) # Gaussian filter <- Matlab: fspecial('gaussian',7,7/6)
    MSCN_window = MSCN_window / sum(sum(MSCN_window))
    mu = cv2.filter2D(Dg, MSCN_window, 'replicate') # ****************replicate 알아오기*********************
    mu_sq = mu * mu
    sigma = math.sqrt(abs(cv2.filter2D(Dg*Dg, MSCN_window, 'replicate') - mu_sq))
    D_MSCN = (Dg - mu)/(sigma+1)
    cv = sigma / mu

    mu1 = cv2.filter2D(Rg, MSCN_window, 'replicate')
    mu_sq1 = mu1 * mu1
    sigma1 = math.sqrt(abs(cv2.filter2D(Rg*Rg, MSCN_window, 'replicate') - mu_sq1))
    R_MSCN = (Rg - mu1)/(sigma + 1)
    cv1 = sigma / mu1

    # feature extraction and similarity calculation
    # MSCN similarity
    n, m = np.shape(D_MSCN) # 세로, 가로
    mc_mscn = max()

    # ho, wo = D.shape # Defogged image's shape
    # hc, wc = R.shape # Reference image's shape
    # ratio_orig = ho/wo # 
    # ratio_comp = hc/wc
    # dim = (wc, hc)