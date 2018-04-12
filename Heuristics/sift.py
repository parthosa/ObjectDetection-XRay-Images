import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
from PIL import Image

path = 'data2\\gunCluttered.png'
imgPath = os.path.join(os.getcwd(),path)
img = cv2.imread(imgPath)
print("#### IMG ####")
print(img)

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("#### GRAY ####")
print(gray)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
out = np.zeros(img.shape)
out = cv2.drawKeypoints(gray,kp,out,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print("#### OUT ####")
print(str(out))
#siftImg = Image.fromarray(out, 'RGB')
#cv2.imwrite('sift_keypoints.jpg',out)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(out,cmap = 'gray')
plt.title('SIFT Image'), plt.xticks([]), plt.yticks([])
plt.show()
