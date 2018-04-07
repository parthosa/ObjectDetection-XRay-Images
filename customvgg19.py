from keras.applications import vgg19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier

# img_path = 'B3.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
img_width, img_height = 224, 224
train_data_dir =  "/data/train"
test_data_dir =  "/data/test"
batch_size = 8


model = vgg19.VGG19(weights='imagenet', include_top=True)
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

train_datagen = ImageDataGenerator(
# rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=180)


train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")


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



# features = model.predict(x)
train_labels = train_generator.classes
test_labels = test_generator.classes

model_extractfeatures = Model(input=model.input, output=model.get_layer('fc1').output)
# fc2_features = model_extractfeatures.predict(x)

train_data = model_extractfeatures.predict_generator(train_generator,steps=113,verbose=1)
test_data = model_extractfeatures.predict_generator(test_generator,steps=132,verbose=1)



np.save('/output/train_features',train_data)
np.save('/output/test_features',test_data)
np.save('/output/train_labels',train_labels)
np.save('/output/test_labels',test_labels)

