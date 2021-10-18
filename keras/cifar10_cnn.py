'''
This example is modified from https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

#Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
import os
import argparse
import spell.metrics as metrics

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, dest='epochs', default=50)
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.0001)
parser.add_argument('--conv2_filter', type=int, dest='conv2_filter', default=64)
parser.add_argument('--conv2_kernel', type=int, dest='conv2_kernel', default=3)
parser.add_argument('--dense_layer', type=int, dest='dense_layer', default=512)

parser.add_argument('--dropout_1', type=float, dest='dropout_1', default=0.25)
parser.add_argument('--dropout_2', type=float, dest='dropout_2', default=0.25)
parser.add_argument('--dropout_3', type=float, dest='dropout_3', default=0.5)
args = parser.parse_args()

batch_size = 32
num_classes = 10
epochs = args.epochs
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(args.dropout_1))

model.add(Conv2D(args.conv2_filter, (args.conv2_kernel, args.conv2_kernel), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(args.conv2_filter, (args.conv2_kernel, args.conv2_kernel)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(args.dropout_2))

model.add(Flatten())
model.add(Dense(args.dense_layer))
model.add(Activation('relu'))
model.add(Dropout(args.dropout_3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = optimizers.RMSprop(lr=args.learning_rate, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        # NOTE added to fix crash (https://github.com/keras-team/keras/issues/11874)
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save trained model.
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_json = model.to_json()
model_json_path = os.path.join(save_dir, "model.json")
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)
print('Saved model json at %s ' % model_json_path)
model_weights_path = os.path.join(save_dir, model_name)
model.save(model_weights_path)
print('Saved trained model weights at %s ' % model_weights_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
