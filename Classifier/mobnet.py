from keras.applications.mobilenet import MobileNet
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.callbacks import CSVLogger, TensorBoard
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.initializers import VarianceScaling

#####load data#####
img_width, img_height = 224, 224
train_data_dir =  "/data/train"
test_data_dir =  "/data/test"
batch_size = 64

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
model = MobileNet(weights=None, classes=4)

#####Compile model#####
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
rms = RMSprop()
# mse = losses.mean_squared_error
adam = Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fit the model
# f = open("output/log.csv","w+")
# csv_logger = CSVLogger('/output/log.csv', append=True, separator=',')
tb_callback = TensorBoard(log_dir='/output/logs', histogram_freq=0, batch_size=batch_size)
# model.fit(train_features, train_labels, epochs=128, batch_size=batch_size,  verbose=2, callbacks=[csv_logger])
model.fit_generator(train_generator, epochs=150, steps_per_epoch = (1400/batch_size)+1, verbose=2, callbacks=[tb_callback], shuffle=True, validation_split=0.1)
model.save("/output/mobnet.h5")
score,acc = model.evaluate_generator(test_generator, steps = (550/batch_size)+1)
# calculate predictions
# pred = model.predict(test_features)
# print(test_labels.shape,pred.shape)
# print(test_labels[0],pred[0])
target_names = ['blade','gun','others','shuriken']
print("Test score: " + str(score))
print("Test accuracy: " + str(acc))


# print(classification_report(test_labels, pred,target_names=target_names))
