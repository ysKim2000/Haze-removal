import cv2
import sys
import numpy as np

# https://deep-learning-study.tistory.com/240

if __name__ == "__main__":
    src = cv2.imread('data/groot.jpg')
    if src is None:
        print('Image load failed!')
        sys.exit()
    rc = cv2.selectROI(src)  # 초기 위치 지정하고 모서리 좌표 4개를 튜플값으로 반환
    mask = np.zeros(src.shape[:2], np.uint16)  # 마스크는 검정색으로 채워져있고 입력 영상과 동일한 크기

    # 결과를 계속 업데이트 하고 싶으면 bgd, fgd 입력
    cv2.grabCut(src, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # grabCut 자료에서 0,2는 배경, 1,3은 전경입니다.
    # mask == 0 or mask == 2를 만족하면 0으로 설정 아니면 1로 설정합니다
    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint16')

    # np.newaxis로 차원 확장
    dst = src * mask2[:, :, np.newaxis]
    
    mask = mask * 64
    cv2.imshow('mask', mask)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()