from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

neighbors = 13

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	else:
		cv2.normalize(hist, hist)

	return hist.flatten()

def extract_sift(image):

	gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray,None)
	# print des
	return des

base = ".."
train_data_dir = base + "/data/train"
test_data_dir = base + "/data/test"


print("describing images...")
imagePathsTrain = list(paths.list_images(train_data_dir))
imagePathsTest = list(paths.list_images(test_data_dir))




def img_proc(imagePaths):
	rawImages = []
	features = []
	labels = []
	sift_features = []

	# loop over the input images
	for (i, imagePath) in enumerate(imagePaths):
		image = cv2.imread(imagePath)
		label = imagePath.split(os.path.sep)[-2:-1][0]

		pixels = image_to_feature_vector(image)
		hist = extract_color_histogram(image)
		sift_feature = extract_sift(image)
		print(sift_feature.shape,hist.shape)
		# print hist

		rawImages.append(pixels)
		features.append(hist)
		sift_features.append(sift_feature)
		labels.append(label)

		print("processed {}/{}".format(i, len(imagePaths)))

		# rawImages = np.array(rawImages)
		# features = np.array(features)
	sift_features = np.array(sift_features)
	labels = np.array(labels)

	return [sift_features, labels]

if(not os.path.exists('../knn/trainFeat.npy')):
	trainFeat ,trainLabels = img_proc(imagePathsTrain)
	np.save('../knn/trainFeat',trainFeat)
	np.save('../knn/trainLabels',trainLabels)
else:
	trainFeat = np.load('../knn/trainFeat.npy')
	trainLabels = np.load('../knn/trainLabels.npy')

if(not os.path.exists('../knn/testFeat.npy')):
	testFeat ,testLabels = img_proc(imagePathsTest)
	np.save('../knn/testFeat',testFeat)
	np.save('../knn/testLabels',testLabels)
else:
	testFeat = np.load('../knn/testFeat.npy')
	testLabels = np.load('../knn/testLabels.npy')

# (trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
# (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(trainFeat, trainLabels, test_size=0.25, random_state=42)
print(trainFeat.shape,trainLabels.shape)
# print("evaluating raw pixel accuracy...")
# model = KNeighborsClassifier(n_neighbors=neighbors)
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("raw pixel accuracy: {:.2f}%".format(acc * 100))

# predicted = model.predict(testFeat)
# report = classification_report(testLabels, predicted)
# print(report)

# # print(trainFeat.shape)
# # trainFeat = np.array(trainFeat).reshape((1, -1))
# # print(trainFeat.shape)
# # print(testFeat.shape)
# # testFeat = np.array(testFeat).reshape((1, -1))
# # print(testFeat.shape)


# # print("evaluating histogram accuracy...")
# # model = KNeighborsClassifier(n_neighbors=neighbors)
# # model.fit(trainFeat, trainLabels)
# # acc = model.score(testFeat, testLabels)
# # print("histogram accuracy: {:.2f}%".format(acc * 100))
# # predicted = model.predict(testFeat)
# # report = classification_report(testLabels, predicted)
# # print(report)