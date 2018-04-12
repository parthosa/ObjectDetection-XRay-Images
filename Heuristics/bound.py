import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
from PIL import Image

class rect(object):
	def __init__(self, x, y, w, h):
		
		# Initializing rect from OpenCV Coordinate system

		self.bx = x
		self.by = y
		self.bw = w
		self.bh = h

	def area(self):
		return self.bh*self.bw

def intersection(ra, rb):

	irx1 = max(ra.bx,rb.bx)
	irx2 = min(ra.bx+ra.bw,rb.bx+rb.bw)
	if irx1>irx2:
		return 0,0
	irw = irx2-irx1
	iry1 = max(ra.by,rb.by)
	iry2 = min(ra.by+ra.bh,rb.by+rb.bh)
	if iry1>iry2:
		return 0,0
	irh = iry2-iry1
	ir = rect(irx1,iry1,irw,irh)
	return ir.area()/ira.area(),ir.area()/irb.area() 




if __name__ == "__main__":

	for j in range(16):
		path = 'Source\\%d.jpg'%(j+1)
		if j==3 or j==2 or j>12:
			path = 'Source\\%d.png'%(j+1)
		imgPath = os.path.join(os.getcwd(),path)
		im = cv2.imread(imgPath)
		# print(im)
		imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		for x in range(4):
			ret,thresh = cv2.threshold(imgray,(x+1)*50,255,0)
			image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			out = np.zeros(im.shape)
			im2 = im.copy()
			#im2 = cv2.drawContours(im2, contours, -1, (0,255,0), 3)
			#npCont = np.array(contours)
			#print(contours[0])
			#print(contours[0].shape)
			#pixelpoints = np.transpose(np.nonzero(out))
			i=0
			print(str(im.shape)+" %03d"%(x+1))
			
			for cnt in contours:
				bx,by,bw,bh = cv2.boundingRect(cnt)
				testImg = im[by-10:by+bh+10,bx-10:bx+bw+10,:]
				if bw>20 and bh>20:
					#print(testImg.shape)
					cv2.imwrite(".\\New\\%d_out_%dth_iter_%03d.jpg"%(j+1,x+1,i), testImg)
					i+=1
					cv2.rectangle(im2,(bx,by),(bx+bw,by+bh),(255,0,0),3)
				#print(testImg)
			cv2.imwrite(".\\New\\Box\\%d_out%d.jpg"%(j+1,x+1), im2)
		

	'''
		plt.subplot(121),plt.imshow(im,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(im2,cmap = 'gray')
		plt.title('Contour Image'), plt.xticks([]), plt.yticks([])
		plt.show()
	'''