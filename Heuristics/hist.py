import cv2
import numpy as np
import os

path = 'data2\\clutteredImg.png'
imgPath = os.path.join(os.getcwd(),path)
im = cv2.imread(imgPath)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(imgray)
res = np.hstack((imgray,equ))

cv2.imwrite('hist.jpg', res)