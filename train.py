
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from model import CCVAE


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))

y_train_cat = to_categorical(y_train).astype(np.float32)
y_test_cat  = to_categorical(y_test).astype(np.float32)
num_classes = y_test_cat.shape[1]


# network parameters
input_shape = x_train[0].shape
lbls_shape = (num_classes,)
batch_size = 256
latent_dim = 10
epochs = 10


ccvae = CCVAE(input_shape, lbls_shape, latent_dim, weights_path='weights.hdf5')
history = ccvae.train(x_train, y_train_cat, batch_size, epochs)
print('Training is done!')