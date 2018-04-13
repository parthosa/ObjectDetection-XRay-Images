from keras.applications.mobilenet import MobileNet
from keras.models import Model, load_model
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import keras
import cv2
import os

from bound import get_bound
from helpers import *

############### Input images #####################
print 'Getting current working directory...'
cwd = os.getcwd()
print 'Loading images...'
images = load_images_from_folder(os.path.join(cwd,"data"))
print '%d Images loaded.' %(len(images))

############### Initiating Model##################
print 'Loading MobileNet model...'
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
	model=load_model(os.path.join(cwd,'model/mobnet.h5'))
print 'Model loaded. Compiling model...'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print 'Model compiled.'
testImg = np.zeros((224,224,3))

############## Main loop ###################
i=0
for img in images:
	#Get bounds
	print '[%d] Getting bounds...' %(i)
	bounds = get_bound(img)
	print '[%d] Bounds retrieved.' %(i)
	predictions  = []
	#Classify bounds
	print '[%d] Classifying bounds...' %(i)
	
	class_bounds = {}
	for bound in bounds:
		bx,by,bw,bh = bound
		# tImg = img[by-10:by+bh+10,bx-10:bx+bw+10,:]
		tImg = img[by:by+bh,bx:bx+bw,:]
		testImg = cv2.resize(tImg, (224, 224), interpolation = cv2.INTER_CUBIC)
		testImg = np.expand_dims(testImg, axis=0)
		predScore = model.predict(testImg, batch_size=1, verbose=0, steps=None)
		prediction = np.argmax(predScore)
		if np.amax(predScore) < 0.75:
			prediction = 2

		try:
			print np.amax(predScore),predScore[prediction]
		except:
			print 'err'

		if prediction!= 2 :
			if get_label(prediction) not in class_bounds:
				class_bounds[get_label(prediction)] = [bound]
			else:
				class_bounds[get_label(prediction)].append(bound)

		# predictions.append(prediction)
	
	print '[%d] Bounds Classified' %(i)

	#Merge bounds

	for label in class_bounds:
		print 'bounds %d in %s' %(len(class_bounds[label]),label)


	print '[%d] Merging bounds...' %(i)
	c = 0;
	for label in class_bounds:
		del_ix = set()
		for ix,bounds in enumerate(class_bounds[label]):
			for j,bounds2 in enumerate(class_bounds[label]):
				if bounds!=bounds2 and j != ix:
					iou=get_iou(modified_bounds(bounds),modified_bounds(bounds2))
					if iou[0] > 0.75 or iou[1] > 0.75:
						c+=1
						if iou[0] > iou[1]:
							del_ix.add(j)
						else:
							del_ix.add(ix)

		# for p,q in enumerate(del_ix):
		# 	del_ix[p]-=p
		del_ix = list(del_ix)
		del_ix.reverse()
		for ix in  del_ix:
			del class_bounds[label][ix]

	print '[%d] Bounds merge: %d' %(i,c)

	for label in class_bounds:
		print 'bounds %d in %s' %(len(class_bounds[label]),label)


	#Print bounds and label on image
	print '[%d] Printing bounds and labels...' %(i)
	k=0
	labels= []

	# for pred in predictions:
	# 	if pred!=2:
	# 		labels.append(get_label(pred))
	# 		k+=1
	# 	else:
	# 		del bounds[k]
			

	img = write_labels(img,class_bounds)
	# img = write_labels(img,labels,bounds)
	print '[%d] %d threats labelled.' %(i,len(bounds))
	#Save
	print '[%d] Saving image...' %(i)
	cv2.imwrite("./output/%d_out.jpg"%(i), img)
	print '[%d] Image saved.' %(i)
	i+=1