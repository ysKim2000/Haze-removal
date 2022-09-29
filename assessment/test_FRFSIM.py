import numpy as np
import cv2

BINS = 256
MAX_LEVEL = BINS - 1

def i2f(img):
    return img / MAX_LEVEL

def f2i(img):
    return np.uint8(np.around(img * MAX_LEVEL))

'''
1. Dark channel features (DF)
2. MSCN features (MF)
3. Gradient features (GF)
4. ChromaHSV features (CF)

5. Fog(FD) 측정(DF*MF)
6. Artifacts(AD) 측정 (GF*CF)

7. 최종 품질 측정(결합) - (FD^{beta1}*AD^{beta2})
'''