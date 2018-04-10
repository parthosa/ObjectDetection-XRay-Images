from keras.applications.mobilenet import MobileNet
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import CustomObjectScope
import keras

#####load data#####
img_width, img_height = 224, 224
test_data_dir =  "../ceeri sop/data/test"
batch_size = 64

test_datagen = ImageDataGenerator(
# rescale = 1./255,
horizontal_flip = False,
fill_mode = "nearest",
zoom_range = 0.0,
width_shift_range = 0.0,
height_shift_range=0.0,
rotation_range=0)

test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

#####get model#####
# model = MobileNet(weights=None, classes=4)

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
	model=load_model('./mobnet.h5')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	score,acc = model.evaluate_generator(test_generator, steps = (550/batch_size)+1)
	# calculate predictions
	# pred = model.predict(test_features)
	# print(test_labels.shape,pred.shape)
	# print(test_labels[0],pred[0])
	target_names = ['blade','gun','others','shuriken']
	print "Test score: %f"%(score)
	print "Test accuracy: %f"%(acc)


# print(classification_report(test_labels, pred,target_names=target_names))