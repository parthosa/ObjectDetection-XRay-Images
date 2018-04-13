import cv2
import os

def load_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			images.append(img)
	return images

def modified_bounds(bounds):
	temp = {}
	temp['x1'] = bounds[0]
	temp['x2'] = bounds[0] + bounds[2]
	temp['y1'] = bounds[1]
	temp['y2'] = bounds[1] + bounds[3]
	return temp

def get_iou(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
	Keys: {'x1', 'x2', 'y1', 'y2'}
	The (x1, y1) position is at the top left corner,
	the (x2, y2) position is at the bottom right corner
	bb2 : dict
	Keys: {'x1', 'x2', 'y1', 'y2'}
	The (x, y) position is at the top left corner,
	the (x2, y2) position is at the bottom right corner

	Returns
	-------
	float
	in [0, 1]
	"""
	# print bb1,bb2
	assert bb1['x1'] < bb1['x2']
	assert bb1['y1'] < bb1['y2']
	assert bb2['x1'] < bb2['x2']
	assert bb2['y1'] < bb2['y2']

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['x1'], bb2['x1'])
	y_top = max(bb1['y1'], bb2['y1'])
	x_right = min(bb1['x2'], bb2['x2'])
	y_bottom = min(bb1['y2'], bb2['y2'])

	if x_right < x_left or y_bottom < y_top:
		return [0,0,0]

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	# compute the area of both AABBs
	bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
	bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = []
	iou.append(float(intersection_area) / float(bb1_area))
	iou.append(float(intersection_area) / float(bb2_area))
	iou.append(float(intersection_area) / float(bb1_area+bb2_area-intersection_area))

	try:
		assert iou[0] >= 0.0
		assert iou[0] <= 1.0
		assert iou[1] >= 0.0
		assert iou[1] <= 1.0
		assert iou[2] >= 0.0
		assert iou[2] <= 1.0
	except:
		print intersection_area, bb1_area, bb2_area

	return iou

def get_label(prediction):
	if prediction==0:
		return "Blade"
	if prediction==1:
		return "Gun"
	if prediction==2:
		return "Other"
	if prediction==3:
		return "Shuriken"


def write_labels(im, class_bounds):
	i = 0
	for label in class_bounds:
		for cord in class_bounds[label]:
			# cord = cords[i]
			bx,by,bw,bh = cord
			font = cv2.FONT_HERSHEY_SIMPLEX
			im = cv2.putText(im,label,(bx+10,by+50), font, 1,(255,0,0),2,cv2.LINE_AA)
			cv2.rectangle(im,(bx,by),(bx+bw,by+bh),(255,0,0),3)
		# i+=1   
	return im