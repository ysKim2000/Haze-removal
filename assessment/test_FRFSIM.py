'''
1. Dark channel features (DS)
2. MSCN features (MS)
3. Gradient features (GS)
4. ChromaHSV features (CS)

5. Fog(FD) 측정(DS*MS)
6. Artifacts(AD) 측정 (GS*CS)

7. 최종 품질 측정(결합) - (FD^{beta1}*AD^{beta2})
'''
import numpy as np
import cv2
from os import path as ospath
from regex import P
from sympy import Derivative
import math

BINS = 256
MAX_LEVEL = BINS - 1

# Parameter setting
K1 = 0.0001
K2 = 0.00005
K3 = 0.00045
K4 = 0.0009
L = 255 # pixel value


def i2f(img):
    return img / MAX_LEVEL

def f2i(img):
    return np.uint8(np.around(img * MAX_LEVEL))

def C(K):
    return K * L


# 1. dark channel similarity
def DS(R, D):
    # Reference image dark channel
    Rdc = np.min(i2f(R), 2)
    # Defogged image dark channel
    Ddc = np.min(i2f(D), 2)
    
    CD = C(K1)
    
    # Dark channel similarity
    SD = (2 * Ddc * Rdc + CD) / (Rdc*Rdc + Ddc*Ddc + CD)
    return np.mean(SD), SD


# # 2. MSCN similarity
def MS(R, D):
    # m, n = np.shape(R)
    # Reference image MSCN, Defogged image MSCN
    R_MSCN, D_MSCN = MSCN(R, D)
    
    # mc_mscn = max(np.var(D_MSCN, R_MSCN))
    # w_mscn = mc_mscn / sum(mc_mscn)
    
    CM = C(K2)
    
    # MSCN similarity
    SM = (2 * R_MSCN * D_MSCN + CM)/(R_MSCN*R_MSCN + D_MSCN*D_MSCN + CM)
    return np.mean(SM)

def MSCN(R, D):
    
    Dg = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY) / MAX_LEVEL
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY) / MAX_LEVEL
    MSCN_window = cv2.getGaussianKernel(7, 7/6)     # Gaussian filter <- Matlab: fspecial('gaussian',7,7/6)
    MSCN_window = MSCN_window / sum(sum(MSCN_window))
    mu = cv2.filter2D(Dg, -1, MSCN_window) 
    mu_sq = mu * mu
    sigma = math.sqrt(abs(cv2.filter2D(Dg*Dg, -1, MSCN_window) - mu_sq))
    D_MSCN = (Dg - mu)/(sigma + 1)
    # cv = sigma / mu

    mu1 = cv2.filter2D(Rg, -1, MSCN_window)
    mu_sq1 = mu1 * mu1
    sigma1 = math.sqrt(abs(cv2.filter2D(Rg*Rg, -1, MSCN_window) - mu_sq1))
    R_MSCN = (Rg - mu1)/(sigma1 + 1)
    # cv1 = sigma / mu1
    return R_MSCN, D_MSCN
    

# 3. gradient similarity
def GS(R, D):
    R_G = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY) / MAX_LEVEL
    D_G = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY) / MAX_LEVEL
    
    # Reference image gradient
    R_gradient = np.gradient( R_G )
    # Defogged image gradient
    D_gradient = np.gradient( D_G )
    
    R_gradient = R_gradient[1]
    D_gradient = D_gradient[1]
    
    CG = C(K3)
    SG = (2*R_gradient*D_gradient+CG)/(D_gradient*D_gradient + R_gradient*R_gradient + CG)
    return np.mean(SG), SG


# 4. color similarity
def CS(R, D):
    # reference image chroma_hsv
    R_color = get_saturation(R) * get_value(R)
    # defogged image chroma_hsv
    D_color = get_saturation(D) * get_value(D)

    CC = C(K4)
    SC = (2*R_color*D_color+CC)/(R_color*R_color + D_color*D_color + CC)
    return np.mean(SC), SC

def get_value(img):
    return np.average(img, 2)

def get_saturation(img):
    return 1.0 - np.min(img, 2) / (np.average(img, 2) + np.finfo(np.float32).eps)


def FRFSIM(R, D):
    # similarity features
    mean_SD, SD = DS(R, D)
    # mean_SM, SM = MS(R, D)
    mean_SG, SG = GS(R, D)
    mean_SC, SC = CS(R, D)
    
    # # TEST
    # print(mean_SD)
    # # print(mean_SM)
    # print(mean_SG)
    # print(mean_SC)
    
    if 0.85 < mean_SD and mean_SD <=1:
        beta1 = 0.2
        beta2 = 0.8
    elif 0 <= mean_SD and mean_SD <= 0.85:
        beta1 = 0.8
        beta2 = 0.2
    else:
        beta1 = 0.5
        beta2 = 0.5        
    # S_FD = SD * SM
    S_FD = SD
    S_AD = SG * SC
    # FRFSIM_MAP = (S_FD ** beta1) * (S_AD ** beta2)
    FRFSIM_MAP = (np.sign(S_FD) * (np.abs(S_FD)) ** (beta1)) * (np.sign(S_AD) * (np.abs(S_AD)) ** (beta2))
    FRFSIM_VALUE = abs(np.mean(FRFSIM_MAP))
    print(FRFSIM_VALUE)
    
if __name__ == "__main__":
    DIR = 'outputs'

    R_PATH = ospath.join('assessment', 'images', '0001-0.jpg') # reference image
    D_PATH = ospath.join('assessment', 'images', '0001-3.jpg') # target image
    
    R = cv2.imread(R_PATH)
    D = cv2.imread(D_PATH)

    FRFSIM(R, D)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
