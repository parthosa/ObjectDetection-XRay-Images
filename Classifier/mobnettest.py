from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#####load data#####
img_width, img_height = 224, 224
test_data_dir =  "/data/test"
batch_size = 16

test_datagen = ImageDataGenerator(
# rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=180)

test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

#####get model#####
model = MobileNet(weights=None, classes=4)
model.load_weights('/model/mobnet.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score,acc = model.evaluate_generator(test_generator, steps = (1050/batch_size)+1)
# calculate predictions
# pred = model.predict(test_features)
# print(test_labels.shape,pred.shape)
# print(test_labels[0],pred[0])
target_names = ['blade','gun','others','shuriken']
print("Test score: " + str(score))
print("Test accuracy: " + str(acc))


# print(classification_report(test_labels, pred,target_names=target_names))