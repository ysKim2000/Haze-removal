from tkinter.tix import MAX
import cv2 
import numpy as np
import copy
from os import path as ospath

BINS = 256
MAX_LEVEL = BINS - 1

DEFAULT_BETA = 1 # Dehazing strength


def quantization(pixels, min, max):
    v = (max - min) / MAX_LEVEL
    return (pixels - min) / v

# g_img: guide image, K: kernel size
def guided_filter(img, g_img, K=(50, 50)):
    # f: filtered
    
    f_i = cv2.blur(img, K)
    f_g = cv2.blur(g_img, K)

    f_ig = cv2.blur(img * g_img, K)
    f_gg = cv2.blur(g_img * g_img, K)

    cov_ig = f_ig - f_i * f_g
    var_gg = f_gg - f_g * f_g

    a = cov_ig / (var_gg + np.finfo(float).eps)
    b = f_i - (a * f_g)

    f_a = cv2.blur(a, K)
    f_b = cv2.blur(b, K)

    return f_a * g_img + f_b


 # First, calculating the depth map
def _calculate_DepthMap(hsv):
    THETA0 = 0.121779
    THETA1 = 0.959710
    THETA2 = -0.780245
    SIGMA = 0.041337

    depth_map = THETA0 + THETA1 * hsv[..., 2] + THETA2 * hsv[..., 1] + np.random.normal(0, SIGMA, hsv[..., 0].shape)
    return depth_map

 # Second, calculating the min-filtered depth map
def _filter_DepthMap(depth_map):
    K = 5 # Size of neighbourhood considered for min filter

    h, w = depth_map.shape[:2]

    relu = lambda x: x if x > 0 else 0
    reverse_relu = lambda bound, x: bound if x > bound else x
    filtered_depth_map = copy.deepcopy(depth_map)
    for i in range(h):
        for j in range(w):
            x_low = relu(i - K)
            x_high =  reverse_relu(h - 1, i + K) + 1
            y_low = relu(j - K)
            y_high =  reverse_relu(w - 1, j + K) + 1
            filtered_depth_map[i][j] = np.min(depth_map[x_low:x_high, y_low:y_high])

    return filtered_depth_map

# Finally, refining the depth map
def _refine_DepthMap(filtered_depth_map, depth_map):
    refined_depth_map = guided_filter(filtered_depth_map, depth_map)
    return refined_depth_map

# value: Intensity
def _estimate_AtmosphericLight(img, value, depth_map):
    depth_map_1d = np.ravel(depth_map)
    rankings = np.argsort(depth_map_1d)

    threshold = 99.9 * len(rankings) / 100

    # Find the indices of array elements that are non-zero, grouped by element.
    indices = np.argwhere(rankings > threshold).ravel()
    
    w = img.shape[1]
    indices_rows = indices // w
    indices_columns = indices % w

    A = np.zeros(3)
    v = -np.inf
    for x in range(len(indices_rows)):
        i = indices_rows[x]
        j = indices_columns[x]
        
        if value[i][j] >= v:
            A = img[i][j]
            v = value[i][j]

    return A

# A: atmospheric light, beta: dehazing strength
def _recover(img, depth_map, A, beta):
    t = np.exp(-beta * depth_map)
    t = np.clip(t, 0.1, 0.9)

    J = np.zeros(img.shape, np.float64)

    for i in range(3):
        J[..., i] = (img[..., i] - A[i].astype("float")) / t + A[i]
        J[..., i] = quantization(J[..., i], np.min(J[..., i]), np.max(J[..., i]))

    return J.astype(np.uint8)


# Color Attenuation Prior
def CAP(img, beta=DEFAULT_BETA, is_only_result=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) / MAX_LEVEL

    depth_map = _calculate_DepthMap(hsv)
    filtered_depth_map = _filter_DepthMap(depth_map)
    refined_depth_map = _refine_DepthMap(filtered_depth_map, depth_map)

    # A = _estimate_AtmosphericLight(img, hsv[..., 2], depth_map)
    A, phase = white_balance(img, hsv, depth_map)
    J = _recover(img, refined_depth_map, A, beta)

    if is_only_result:
        return J
    else:
        depth_map = quantization(depth_map, depth_map.min(), depth_map.max())
        filtered_depth_map = quantization(filtered_depth_map, filtered_depth_map.min(), filtered_depth_map.max())
        refined_depth_map = quantization(refined_depth_map, refined_depth_map.min(), refined_depth_map.max())
        return (J, depth_map.astype(np.uint8), filtered_depth_map.astype(np.uint8), refined_depth_map.astype(np.uint8), phase)


def white_balance(img, hsv, depth_map):
    A = _estimate_AtmosphericLight(img, hsv[..., 2], depth_map)

    # Compute white balanced A
    I_WB = gray_world(img)
    A_WB = _estimate_AtmosphericLight(I_WB, hsv[..., 2], depth_map)
    EPSILON = 0.02

    if np.max(A) - np.min(A) < np.max(A_WB) - np.min(A_WB) + EPSILON:
        print('Phase I - Normal') 
        phase = "normal_"
    else:
        print('Phase II - White Balance')
        phase = "wb_"
        A = A_WB
        
    return A, phase

def gray_world(img):
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

import os

if __name__ == "__main__":
    DIR = 'dehaze/outputs'
    # file_name = 'GSGL_Bing_681.png'
    # I_PATH = ospath.join('./data/hazy', file_name)

    # I = cv2.imread(I_PATH)
    # J, depth_map, filtered_depth_map, refined_depth_map, phase = CAP(I, is_only_result=False)
    # O_PATH = ospath.join(DIR, "cap_"+phase+file_name)
    
    def print_files_in_dir(root_dir):
        files = os.listdir(root_dir)
        for file in files:
            path = os.path.join(root_dir, file)
            I = cv2.imread(path)
            J, depth_map, filtered_depth_map, refined_depth_map, phase = CAP(I, is_only_result=False)
            NORMAL_PATH = ospath.join('dehaze/outputs/normal', "cap_"+phase+file)
            WB_PATH = ospath.join('dehaze/outputs/white balance', "cap_"+phase+file)
            # O_PATH = ospath.join(DIR, "cap_"+phase+file)
            if(phase == "wb_"):
                cv2.imwrite(WB_PATH, J)
            else:
                cv2.imwrite(NORMAL_PATH, J)
            
    print_files_in_dir("C:/Users/ys/Desktop/RTTS/JPEGImages",)
    
    
    # if(phase == "wb_"):
    #     cv2.imwrite(O_PATH, J)
    # cv2.imshow('Haze image', I)
    # cv2.imshow('Dehazed image', J / MAX_LEVEL)
    cv2.waitKey()

    # cv2.imwrite(ospath.join(DIR, 'depth map (CAP).jpg'), depth_map)
    # cv2.imwrite(ospath.join(DIR, 'min-filtered depth map (CAP).jpg'), filtered_depth_map)
    # cv2.imwrite(ospath.join(DIR, 'refined depth map (CAP).jpg'), refined_depth_map)
    
