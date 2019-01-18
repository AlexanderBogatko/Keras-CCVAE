
from model import CCVAE

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