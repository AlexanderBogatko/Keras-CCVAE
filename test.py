from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from model import CCVAE


def draw_digits(args):
	digit_size = 28
	args = [x.squeeze() for x in args]
	n = min([x.shape[0] for x in args])
	figure = np.zeros((digit_size * len(args), digit_size * n))

	for i in range(n):
	    for j in range(len(args)):
	        figure[j * digit_size: (j + 1) * digit_size,
	               i * digit_size: (i + 1) * digit_size] = args[j][i].squeeze()

	plt.figure(figsize=(2*n, 2*len(args)), num='Style Transfer')
	plt.imshow(figure, cmap='Greys_r')
	plt.grid(False)
	ax = plt.gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.show()

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
input_shape = (28, 28, 1)
lbls_shape = (10,)
batch_size = 256
latent_dim = 10
epochs = 10
digit = 8

ccvae = CCVAE(input_shape, lbls_shape, latent_dim, weights_path='weights.hdf5')

# draw manifold of digit
ccvae.draw_manifold(digit)

# style transfer
n = 10					# num of prototypes
lbl = 5					# digit to transfer style
generated = []
prototypes = x_test[y_test == lbl][:n]

for i in range(num_classes):
    generated.append(ccvae.style_transfer(prototypes, lbl, i))

draw_digits(generated)