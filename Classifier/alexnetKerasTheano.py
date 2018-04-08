import os    
from keras import backend as K
from theano import tensor as T
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.initializers import VarianceScaling
from keras.callbacks import CSVLogger
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from customlayers import convolution2Dgroup, crosschannelnormalization, splittensor, Softmax4D

os.environ['THEANO_FLAGS'] = "device=gpu"   

# import sys
# sys.path.insert(0, '../convnets-keras')



def mean_subtract(img):   
	img = T.set_subtensor(img[:,0,:,:],img[:,0,:,:] - 123.68)
	img = T.set_subtensor(img[:,1,:,:],img[:,1,:,:] - 116.779)
	img = T.set_subtensor(img[:,2,:,:],img[:,2,:,:] - 103.939)

	return img / 255.0

def get_alexnet(input_shape,nb_classes,mean_flag): 
	# code adapted from https://github.com/heuritech/convnets-keras

	inputs = Input(shape=input_shape)

	if mean_flag:
		mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(inputs)
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
			name='conv_1', init='he_normal')(mean_subtraction)
	else:
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
			name='conv_1', init='he_normal')(inputs)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2 = merge([
	    Convolution2D(128,5,5,activation="relu",init='he_normal', name='conv_2_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_2)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1))(conv_3)
	conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3',init='he_normal')(conv_3)

	conv_4 = ZeroPadding2D((1,1))(conv_3)
	conv_4 = merge([
	    Convolution2D(192,3,3,activation="relu", init='he_normal', name='conv_4_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_4)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

	conv_5 = ZeroPadding2D((1,1))(conv_4)
	conv_5 = merge([
	    Convolution2D(128,3,3,activation="relu",init='he_normal', name='conv_5_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_5)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

	dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

	dense_1 = Flatten(name="flatten")(dense_1)
	dense_1 = Dense(4096, activation='relu',name='dense_1',init='he_normal')(dense_1)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096, activation='relu',name='dense_2',init='he_normal')(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(nb_classes,name='dense_3_new',init='he_normal')(dense_3)

	prediction = Activation("softmax",name="softmax")(dense_3)

	alexnet = Model(input=inputs, output=prediction)
    
	return alexnet


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

#####load data#####
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

#####get model#####
net = get_alexnet((227, 227, 3), 4, False)

#####Compile model#####
sgd = SGD(lr=0.000001, decay=1e-6, momentum=1.9)
rms = RMSprop()
mse = losses.mean_squared_error

net.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

# Fit the net
# f = open("output/log.csv","w+")
csv_logger = CSVLogger('/output/log.csv', append=True, separator=',')
# net.fit(train_features, train_labels, epochs=128, batch_size=batch_size,  verbose=2, callbacks=[csv_logger])
net.fit_generator(train_generator, epochs=50, steps_per_epoch = (900/batch_size)+1, verbose=2, callbacks=[csv_logger])
net.save_weights("/output/alexnetKerasTheano.h5")
score,acc = net.evaluate_generator(test_generators)
# calculate predictions
# pred = net.predict(test_features)
# print(test_labels.shape,pred.shape)
# print(test_labels[0],pred[0])
target_names = ['blade','gun','others','shuriken']
print("Test score: %f"(score))
print("Test accuracy: %f"(acc))


# print(classification_report(test_labels, pred,target_names=target_names))