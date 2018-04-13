import cv2
import numpy as np

def get_bound(im):
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	bound = []
	w,h,c = im.shape
	lim = min(w,h)*0.05
	for x in range(4):
		ret,thresh = cv2.threshold(imgray,(x+1)*50,255,0)
		image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			bx,by,bw,bh = cv2.boundingRect(cnt)
			if bw>lim and bh>lim:
				bound.append((bx,by,bw,bh))
	return bound