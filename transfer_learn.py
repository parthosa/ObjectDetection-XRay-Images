import os    
os.environ['THEANO_FLAGS'] = "device=cpu"  

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image
# import preprocess_input
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score
from alexnet import get_alexnet
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

k.set_image_dim_ordering('th')

img_width, img_height = 227, 227
input_size = (3,227,227)
# input_size_vgg = (227,227,3)

base = ".."
train_data_dir = base + "/data/train"
test_data_dir = base + "/data_new/"
nb_train_samples = 900
nb_test_samples = 1050 
batch_size = 8
# epochs = 10
num_classes = 4


alexnet = get_alexnet(input_size,num_classes,True)
alexnet.load_weights(base + "/weights/alexnet_weights.h5", by_name=True)

alexnet = Model(input=alexnet.input, output=alexnet.get_layer('dense_1').output)  
# # weights = np.load('../bvlc_alexnet.npy')
# # alexnet_convolutional_only.set_weights(weights)
# print(alexnet.summary())

# model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = input_size_vgg,pooling='max')

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
# for layer in model.layers[:5]:
#     layer.trainable = False


# model_final = Model(input = model.input, output = model.get_layer())
# print(model.summary())

# img = image.load_img('../data_another_copy/test/blade/B0081_0001.png', target_size=(224, 224)) 
# x = image.img_to_array(img) 
# x = np.expand_dims(x, axis=0) 
# print(x.shape)
# # x = preprocess_input(x)
# features = model.predict(x)
# features2 = model2.predict(x)


# # print(features)
# # print(features2)
# print()


#Adding custom Layers 
# x = model.output
# x = Flatten(name='flatten')(x)
# x = Dense(4096, activation='relu', name='fc1')(x)
# # x = Dropout(0.5)(x)
# # x = Dense(1024, activation="relu")(x)
# # predictions = Dense(num_classes, activation="softmax")(x)

# # # creating the final model 
# model_final = Model(input = model.input, output = x)

# # # compile the model 
# model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
# train_datagen = ImageDataGenerator(
# # rescale = 1./255,
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=180)

test_datagen = ImageDataGenerator(
# rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=180)

# train_generator = train_datagen.flow_from_directory(
# train_data_dir,
# target_size = (img_height, img_width),
# batch_size = batch_size, 
# class_mode = "categorical")

test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = None)


# for X_batch, Y_batch in train_generator:
# 	# for x in X_batch:
# 	# 	x = np.expand_dims(x, axis=0)
# 	pred = model.predict(X_batch)
# 	print(pred.shape)
# 	print(X_batch.shape,Y_batch.shape)


# # Save the model according to the conditions  
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
# k.clear_session()
# train_labels = train_generator.classes
test_labels = test_generator.classes
test_names = list(test_generator.class_indices.keys())

# print(model_final.summary())
# train_data = alexnet.predict_generator(train_generator,steps=113,verbose=1)
# print(train_data.shape)
# np.save(base + '/output/train_features',train_data)
# np.save(base + '/output/train_labels',train_labels)

test_data = alexnet.predict_generator(test_generator,steps=33,verbose=1)
print(test_data.shape)
np.save(base + '/output/test_features',test_data)
np.save(base + '/output/test_labels',test_labels)
np.save(base + '/output/test_names',test_names)
 

# # # Train the model 
# # model_final.fit_generator(
# # train_generator,
# # samples_per_epoch = nb_train_samples,
# # epochs = epochs,
# # test_data = test_generator,
# # nb_val_samples = nb_test_samples,
# # callbacks = [checkpoint, early])

# # # model_final.save