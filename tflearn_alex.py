from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.optimizers import Momentum
import numpy as np
import pickle

from tflearn.data_utils import image_preloader,build_image_dataset_from_dir
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import os
import tensorflow as tf

# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Building 'AlexNet'



base = ''
X_train, Y_train = build_image_dataset_from_dir(base + '/data/train',
                                        dataset_file= base + '/model1/train.pkl',
                                        resize=(227,227),
                                        convert_gray=False,
                                        shuffle_data=False,
                                        categorical_Y=True)

X_test, Y_test = build_image_dataset_from_dir(base + '/data/test',
                                        dataset_file= base + '/model1/test.pkl',
                                        resize=(227,227),
                                        convert_gray=False,
                                        shuffle_data=False,
                                        categorical_Y=True)

X_train = np.expand_dims(X_train,axis=3)
X_train = np.repeat(X_train,3,axis=3)

X_test = np.expand_dims(X_test,axis=3)
X_test = np.repeat(X_test,3,axis=3)

init = np.load(base + "/pretrained/alexnet.npz",encoding = 'latin1').item()

img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center()

imgaug = tflearn.ImageAugmentation()
imgaug.add_random_rotation (max_angle=360.0)

# network = input_data(shape=[None, 227, 227, 1])
# network = conv_2d(network, 96, 11, strides=4, activation='relu')
# network = max_pool_2d(network, 3, strides=2)
# network = local_response_normalization(network)
# network = conv_2d(network, 256, 5, activation='relu')
# network = max_pool_2d(network, 3, strides=2)
# network = local_response_normalization(network)
# network = conv_2d(network, 384, 3, activation='relu')
# network = conv_2d(network, 384, 3, activation='relu')
# network = conv_2d(network, 256, 3, activation='relu')
# network = max_pool_2d(network, 3, strides=2)
# network = local_response_normalization(network)
# network = fully_connected(network, 4096, activation=None)
network = input_data(shape=[None, 227, 227,3],data_augmentation=imgaug)
network = conv_2d(network, 96, 11, strides=4,weights_init=tf.constant_initializer(init['conv1']['weights']),bias_init=tf.constant_initializer(init['conv1']['biases']), activation='relu')
network = local_response_normalization(network)
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 256, 5,weights_init=tf.constant_initializer(init['conv2']['weights']),bias_init=tf.constant_initializer(init['conv2']['biases']), activation='relu')
network = local_response_normalization(network)
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 384, 3,weights_init=tf.constant_initializer(init['conv3']['weights']),bias_init=tf.constant_initializer(init['conv3']['biases']), activation='relu')
network = conv_2d(network, 384, 3,weights_init=tf.constant_initializer(init['conv4']['weights']),bias_init=tf.constant_initializer(init['conv4']['biases']), activation='relu')
network = conv_2d(network, 256, 3,weights_init=tf.constant_initializer(init['conv5']['weights']),bias_init=tf.constant_initializer(init['conv5']['biases']), activation='relu')
network = max_pool_2d(network, 3, strides=2)
# network = local_response_normalization(network)
network = fully_connected(network, 4096,weights_init=tf.constant_initializer(init['fc6']['weights']),bias_init=tf.constant_initializer(init['fc6']['biases']) ,activation=None)
# network = dropout(network, 0.5)
# network = fully_connected(network, 4096, activation='tanh')
# network = dropout(network, 0.5)
# network = fully_connected(network, 17, activation='softmax')
# network = regression(network, optimizer='momentum',
#                      loss='categorical_crossentropy',
#                      learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)

train_features = model.predict(X_train)
np.save(base + '/output/train_features',train_features)

test_features = model.predict(X_test)
np.save(base + '/output/test_features',test_features)


# train_features = model.predict(X_train)
np.save(base + '/output/train_labels',Y_train)
np.save(base + '/output/test_labels',Y_test)

# model.load('../weights/alexnet_weights.h5')
# model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
#           show_metric=True, batch_size=64, snapshot_step=200,
#           snapshot_epoch=False, run_id='alexnet_oxflowers17')