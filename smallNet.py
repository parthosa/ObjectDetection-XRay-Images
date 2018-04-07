from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import classification_report,accuracy_score
from keras.initializers import VarianceScaling
from keras.callbacks import CSVLogger


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


# fix random seed for reproducibility
np.random.seed(7)

train_features = sq_norm(np.load('/model/test_features.npy'))

train_labels = get_one_hot(np.load('/model/test_labels.npy'))

test_features = sq_norm(np.load('/model/train_features.npy'))

test_labels = get_one_hot(np.load('/model/train_labels.npy'))

print(train_features.shape)
print(test_features.shape)
print(train_labels.shape)
print(test_labels.shape)

#load data
# encode class values as integers

# create model
model = Sequential()
model.add(Dense(4096, input_shape=(4096,), activation='relu', name='fc1', kernel_initializer=VarianceScaling()))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu', kernel_initializer=VarianceScaling()))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_initializer=VarianceScaling()))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax', kernel_initializer=VarianceScaling()))
# model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
# f = open("output/log.csv","w+")
csv_logger = CSVLogger('/output/log.csv', append=True, separator=',')
model.fit(train_features, train_labels, epochs=150, batch_size=100,  verbose=2, callbacks=[csv_logger])
model.save_weights("/output/model.h5")
scores = model.evaluate(test_features, test_labels, verbose=0)
# calculate predictions
# pred = model.predict(test_features)
# print(test_labels.shape,pred.shape)
# print(test_labels[0],pred[0])
target_names = ['blade','gun','others','shuriken']
print(scores[1])

# print(classification_report(test_labels, pred,target_names=target_names))
