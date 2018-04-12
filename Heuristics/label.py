import cv2
import os

def writeLabel(im, label, path):	

	font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(im,label,(10,50), font, 1,(255,0,0),2,cv2.LINE_AA)
    
    try:
	    cv2.imwrite(path,img)
		return True
    except Exception as e:
    	return False