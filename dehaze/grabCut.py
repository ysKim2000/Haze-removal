import cv2
import numpy as np

# https://deep-learning-study.tistory.com/240 

if __name__ == "__main__":
    I = cv2.imread('data/haze_image.png')
    # hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # print(v)
    # print(type(v))
    # print(np.shape(v))
    cv2.imshow('test',I)
    cv2.waitKey()