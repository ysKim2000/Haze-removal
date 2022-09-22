import numpy as np
import math
import numpy.matlib
import cv2

BINS = 256
MAX_LEVEL = BINS - 1


D = cv2.imread('assessment/data/sbte.jpg') # Defogged image
R = cv2.imread('assessment/data/GRCN.png') # Reference image

# Basic SETUP OF FRFSIM
radius = 1.5
# d = np.diff(getrangefromclass(D))
d = [0, 255]
C = [(0.01*d)^2, ((0.01*d)^2)/2, ((0.03*d)^2)/2, (0.03*d)^2]
    
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
MSCN_window = cv2.getGaussianKernel(7, 7/6)     # Gaussian filter <- Matlab: fspecial('gaussian',7,7/6)
MSCN_window = MSCN_window / sum(sum(MSCN_window))
mu = cv2.filter2D(Dg, MSCN_window, 'replicate') # ****************replicate 알아오기*********************
mu_sq = mu * mu
sigma = math.sqrt(abs(cv2.filter2D(Dg*Dg, MSCN_window, 'replicate') - mu_sq))
D_MSCN = (Dg - mu)/(sigma + 1)
cv = sigma / mu

mu1 = cv2.filter2D(Rg, MSCN_window, 'replicate')
mu_sq1 = mu1 * mu1
sigma1 = math.sqrt(abs(cv2.filter2D(Rg*Rg, MSCN_window, 'replicate') - mu_sq1))
R_MSCN = (Rg - mu1)/(sigma + 1)
cv1 = sigma / mu1

# feature extraction and similarity calculation
# MSCN similarity
# n, m = np.shape(D_MSCN) # 세로, 가로
mc_mscn = max(np.var(D_MSCN, R_MSCN))
w_mscn = mc_mscn / sum(mc_mscn[:])
# mc_mscn = numpy.matlib.repmat(mc_mscn,m,1)
# CM = d / mc_mscn
CM = C(1)
SM = (2*D_MSCN*R_MSCN+CM)/(D_MSCN*D_MSCN+R_MSCN*R_MSCN)
mean_SM = np.mean(SM[:])

# dark channel similarity
Drn = D_R / 255 # Defogged image dark channel
Dgn = D_G / 255
Dbn = D_B / 255
Ddc = min(min(Drn, Dgn), Dbn)
    
Rrn = R_R / 255 # Reference image dark channel
Rgn = R_G / 255
Rbn = R_B / 255
Rdc = min(min(Rrn,Rgn), Rbn)

CD = C(2)
SD = (2*Ddc*Rdc+CD)/(Ddc*Ddc + Rdc*Rdc+CD)
mean_SD = np.mean(SD[:])

# color similarity
D_color = s_d*v_d # Defogged image chroma_hsv
R_color = s_r*v_r # reference image chroma_hsv

CC = C(3)
SC = (2*D_color * R_color + CC) / (D_color*D_color+R_color*R_color+CC)
mean_SC = np.mean(SC[:])

# gradient similarity
D_gradient = np.gradient(Dg) # Defogged image gradient
R_gradient = np.gradient(Rg) # reference image gradient

CG = C(4)
SG = (2*D_gradient*R_gradient+CG)/(D_gradient*D_gradient+R_gradient*R_gradient+CG)
mean_SG = np.mean(SG[:])

# pooling
if 0.85 < mean_SD and mean_SD <=1:
    b1 = 0.2
    b2 = 0.8
elif 0 <= mean_SD and mean_SD <= 0.85:
    b1 = 0.8
    b2 = 0.2
else:
    b1 = 0.5
    b2 = 0.5   

frfsimmap = ((SM*SD)^b1)*((SG*SC)^b2)
frfsmval = abs(np.mean(frfsimmap[:]))

# function gaussFilt = getGaussianWeightingFilter(radius,N)

# Get 2D or 3D Gaussianweighting filter

filtRadius = math.ceil(radius*3) # Standard deviations include >99% of the area.
filtSize = 2*filtRadius + 1

if N < 3:
    # 2D Gaussian mask can be used for filtering even one-dimensional
    # signals using imfilter.
    gaussFilt = cv2.getGaussianKernel([filtSize, filtSize], radius)
else:
    # 3D Gaussian mask
    #  [x,y,z] = np.ndgrid(-filtRadius:filtRadius,-filtRadius:filtRadius, ...-filtRadius:filtRadius)
    # arg = -(x*x + y*y + z*z)/(2*radius*radius)
    gaussFilt = np.exp(arg)
    # gaussFilt(gaussFilt < np.eps*max(gaussFilt[:])) = 0
    sumFilt = sum(gaussFilt[:])
    if sumFilt != 0:
         gaussFilt  = gaussFilt/sumFilt