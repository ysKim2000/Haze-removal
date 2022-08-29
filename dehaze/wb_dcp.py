"""
Single Image Haze Removal Using Dark Channel Prior

Original source
https://github.com/He-Zhang/image_dehaze/blob/master/dehaze.py

Modified by Wonvin Kim, 01/04/22
"""

import cv2 
import numpy as np
import math
from os import path as ospath

BINS = 256
MAX_LEVEL = BINS - 1


def i2f(img):
    return img / MAX_LEVEL

def f2i(img):
    return np.uint8(np.around(img * MAX_LEVEL))


# g_img: guide image, K: kernel size
def guided_filter(img, g_img, K):
    # f: filtered

    f_i = cv2.boxFilter(img, cv2.CV_64F, (K, K))
    f_g = cv2.boxFilter(g_img, cv2.CV_64F, (K, K))
  
    f_ig = cv2.boxFilter(img * g_img, cv2.CV_64F, (K, K))
    f_gg = cv2.boxFilter(g_img * g_img, cv2.CV_64F, (K, K))

    cov_ig = f_ig - f_i * f_g
    var_gg = f_gg - f_g * f_g

    a = cov_ig / (var_gg + np.finfo(float).eps)
    b = f_i - a * f_g

    f_a = cv2.boxFilter(a, cv2.CV_64F, (K, K))
    f_b = cv2.boxFilter(b, cv2.CV_64F, (K, K))

    return f_a * g_img + f_b


# K: kernel size
def get_DarkChannel(img, K):
    R = np.min(img, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, K))
    return cv2.erode(R, kernel)

# base_2d: dark channel in dark channel prior scheme
def estimate_AtmosphericLight(img, base_2d):
    h, w = img.shape[:2]
    resolution = h * w

    # n_px: the number of pixels
    n_px = int( max( math.floor(resolution / 1000), 1 ) )
    
    # v_: vector
    v_img = img.reshape(resolution, 3)
    v_base = base_2d.reshape(resolution)

    args = v_base.argsort()[resolution - n_px:]

    sum = np.zeros([3])
    for i in range(n_px):
       sum += v_img[args[i]]

    return sum / n_px

# A: atmospheric light, K: kernel size
def estimate_TransmissionMap(img, A, K):
    omega = 0.95
    R = np.empty(img.shape, img.dtype)

    for i in range(3):
        R[..., i] = img[..., i] / A[i]

    return 1 - omega * get_DarkChannel(R, K)

# map: transmission map
def refine_TransmissionMap(img, map):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / MAX_LEVEL    
    return guided_filter(map, gray, K=60)

# map: transmission map, A: atmospheric light
def recover(img, map, A):
    J = np.zeros(img.shape, img.dtype)
    map = cv2.max(map, 0.1)

    for i in range(3):
        J[..., i] = (img[..., i] - A[i]) / map + A[i]

    return np.clip(J, 0, 1)


# Dark Channel Prior
def DCP(img, is_only_result=True):
    I = i2f(img)

    dark_channel = get_DarkChannel(I, K=15)
    
    # A = estimate_AtmosphericLight(I, dark_channel)
    A, phase = white_balance(I, dark_channel) # <- white balance A or original A
    
    transmission_map = estimate_TransmissionMap(I, A, 15)
    transmission_map = refine_TransmissionMap(img, transmission_map)
    J = recover(I, transmission_map, A)
    J = f2i(J)

    return J if is_only_result else (J, f2i(dark_channel), f2i(transmission_map), phase)

def white_balance(img, dark_channel):
    A = estimate_AtmosphericLight(img, dark_channel)
    
    # Compute white balanced A
    I_WB = gray_world(img)
    A_WB = estimate_AtmosphericLight(I_WB, dark_channel)
    
    EPSILON = 0.02
    
    if np.max(A) - np.min(A) < np.max(A_WB) - np.min(A_WB) + EPSILON:
        print('Phase I - Normal')
        phase = "normal_"
    
    else:
        print('Phase II - White Balance')
        phase = "wb_"
        A = A_WB
        
    # out = np.empty(img.shape, img.dtype)
    # for i in range(3):
    #     out[..., i] = I[..., i] / A[i]
        
    return A, phase


def get_value(img):
    return np.average(img, 2)

# White balance using gray world assumption
def gray_world(img): # gray-world assumption algorithm
    b, g, r = cv2.split(img)

    R = np.empty(img.shape, img.dtype) 

    # r,g,b channeldml 평균값
    mu_r = np.average(r)
    mu_g = np.average(g)
    mu_b = np.average(b)
    
    # white balance 수식
    R[..., 0] = np.minimum(b * (mu_g / mu_b), 1.0) # blue
    R[..., 2] = np.minimum(r * (mu_g / mu_r), 1.0) # red
    R[..., 1] = g # green
 
    return  R

def normalize(img):
    R = np.empty(img.shape, img.dtype) 

    for i in range(3):
        min = img[..., i].min()
        max = img[..., i].max()
        R[..., i] = (img[..., i] - min) / (max - min)
        R[..., i] = np.clip(R[..., i], 0.0, 1.0)     
        
    return R

if __name__ == "__main__":
    DIR = 'dehaze/outputs'
    file_name = 'GRCN.png'
    I_PATH = ospath.join('./data/hazy', file_name)

    I = cv2.imread(I_PATH)
    J, dark_channel, transmission_map, phase = DCP(I, is_only_result=False)
    O_PATH = ospath.join(DIR, "dcp_"+phase+file_name)

    cv2.imwrite(O_PATH, J)
    cv2.imshow('Haze image', I)
    cv2.imshow('Dehazed image', J / MAX_LEVEL)

    # cv2.imwrite(ospath.join(DIR, 'dark channel (DCP).jpg'), dark_channel)
    # cv2.imwrite(ospath.join(DIR, 'transmission map (DCP).jpg'), transmission_map)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
