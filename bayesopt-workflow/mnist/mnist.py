import tensorflow as tf
import idx2numpy
import keras
import argparse
import spell.metrics
import time
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout
from keras.callbacks import LambdaCallback

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int,
                    dest='batch_size', help='minibatch size',
                    default=128)
parser.add_argument('--epochs', type=int,
                    dest='epochs', help='epochs to train for',
                    default=1)

parser.add_argument('--conv1-filters', type=int,
                    dest='c1_filters', help='number of convolution filters in the first convolutional layer',
                    default=20)
parser.add_argument('--conv1-size', type=int,
                    dest='c1_size', help='convolutional 1 filter size (will be N x N)',
                    default=3)

parser.add_argument('--conv2-filters', type=int,
                    dest='c2_filters', help='number of convolution filters in the second convolutional layer',
                    default=50)
parser.add_argument('--conv2-size', type=int,
                    dest='c2_size', help='convolutional 2 filter size (will be N x N)',
                    default=3)

parser.add_argument('--conv3-filters', type=int,
                    dest='c3_filters', help='number of convolution filters in the second convolutional layer',
                    default=50)
parser.add_argument('--conv3-size', type=int,
                    dest='c3_size', help='convolutional 2 filter size (will be N x N)',
                    default=3)

parser.add_argument('--dense-size', type=int,
                    dest='f1_filters', help='size of fully connected layer',
                    default=128)

parser.add_argument('--dropout', type=float,
                    dest='dropout', help='dropout rate',
                    default=.25)

args = parser.parse_args()

x = idx2numpy.convert_from_file('train-images-idx3-ubyte')
y = idx2numpy.convert_from_file('train-labels-idx1-ubyte')

t_x = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
t_y = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

classes = 10

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1).astype('float32') / 255
t_x = t_x.reshape(t_x.shape[0], t_x.shape[1], t_x.shape[2], 1).astype('float32') / 255

y = keras.utils.to_categorical(y, classes)
t_y = keras.utils.to_categorical(t_y, classes)

model = Sequential()
model.add(Conv2D(args.c1_filters, (args.c1_size, args.c1_size), padding="same", activation="relu", input_shape=(x.shape[1], x.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(args.c2_filters, (args.c2_size, args.c2_size), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(args.c3_filters, (args.c3_size, args.c3_size), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(args.f1_filters, activation="relu"))
model.add(Dropout(args.dropout))
model.add(Dense(classes, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

def send_metric(batch, logs):
    spell.metrics.send_metric("another_loss", 1)
    spell.metrics.send_metric("my_loss", float(logs.get('loss')))

batch_loss = LambdaCallback(
    on_batch_end=lambda batch, logs: send_metric(batch, logs))


model.fit(x, y,
		  batch_size=args.batch_size,
		  epochs=args.epochs,
          verbose=1,
          validation_data=(t_x, t_y),
          callbacks=[batch_loss])

score = model.evaluate(t_x, t_y, verbose=0)

spell.metrics.send_metric('test_acc', score[1])
print('Test loss:', score[0])
print('Test accuracy:', score[1])

if score[1] < .98:
     print("Accuracy below threshold (.98), test failed. Exiting with nonzero status.")
     exit(2)
else:
     print("Accuracy above threshold (.98), test succeeded.")
     exit(0)
