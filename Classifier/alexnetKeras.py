from keras.models import Sequential
from keras import losses
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge, Lambda, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# from convnetskeras.customlayers import convolution2Dgroup, BatchNormalization, splittensor, Softmax4D
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import classification_report,accuracy_score
from keras.initializers import VarianceScaling
from keras.callbacks import CSVLogger
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
import numpy as np

def sq_norm(mat):
    normM = mat - mat.min()
    normM = np.sqrt(normM)
    normM = normM / normM.max()
    mat_sq_norm = normM
    return mat_sq_norm

def get_one_hot(Y):
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)
	return dummy_y

#load data
img_width, img_height = 227, 227
train_data_dir =  "/data/train"
test_data_dir =  "/data/test"
batch_size = 128

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



# fix random seed for reproducibility
np.random.seed(7)

# train_features = sq_norm(np.load('/model/train_features.npy'))
# train_labels = get_one_hot(np.load('/model/train_labels.npy'))
# test_features = sq_norm(np.load('/model/test_features.npy'))
# test_labels = get_one_hot(np.load('/model/test_labels.npy'))

# print(train_features.shape)
# print(test_features.shape)
# print(train_labels.shape)
# print(test_labels.shape)


# create model
model = Sequential()
model.add(Convolution2D(96, 11, 11,subsample=(4,4),activation='relu', name='conv_1', init='he_normal', input_shape=(227,227,3)))
model.add(MaxPooling2D((3, 3), strides=(2,2)))
# model.add(BatchNormalization(name="convpool_1"))
model.add(ZeroPadding2D((2,2)))
model.add(Convolution2D(256,5,5,activation="relu",init='he_normal', name='conv_2'))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
# model.add(BatchNormalization())
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(384,3,3,activation='relu',name='conv_3',init='he_normal'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(384,3,3,activation="relu", init='he_normal', name='conv_4'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3,activation="relu",init='he_normal', name='conv_5'))
model.add(MaxPooling2D((3, 3), strides=(2,2),name="convpool_5"))
model.add(Flatten(name="flatten"))
model.add(Dense(4096, activation='relu',name='dense_1',init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu',name='dense_2',init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(4,name='dense_3_new',init='he_normal'))
model.add(Activation("softmax",name="softmax"))


# model.add(Activation('softmax'))

# Compile model
sgd = SGD(lr=0.000001, decay=1e-6, momentum=1.9)
rms = RMSprop()
mse = losses.mean_squared_error

model.compile(loss=mse, optimizer=rms, metrics=['accuracy'])

# Fit the model
# f = open("output/log.csv","w+")
csv_logger = CSVLogger('/output/log.csv', append=True, separator=',')
# model.fit(train_features, train_labels, epochs=128, batch_size=batch_size,  verbose=2, callbacks=[csv_logger])
model.fit_generator(train_generator, epochs=50, steps_per_epoch = (900/batch_size)+1, verbose=2, callbacks=[csv_logger])
model.save_weights("/output/alexnetKeras.h5")
score,acc = model.evaluate_generator(test_generators)
# calculate predictions
# pred = model.predict(test_features)
# print(test_labels.shape,pred.shape)
# print(test_labels[0],pred[0])
target_names = ['blade','gun','others','shuriken']
print("Test score: %f"(score))
print("Test accuracy: %f"(acc))


# print(classification_report(test_labels, pred,target_names=target_names))