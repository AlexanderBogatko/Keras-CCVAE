
from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm

class CCVAE:

    def __init__(self, input_shape, lbls_shape, latent_dim, optimizer='adam', weights_path=''):

        self.digit_size = 28
        self.latent_dim = latent_dim

        ########## Encoder ############

        encoder_input = Input(shape=input_shape)
        enc_lbls_input = Input(shape=lbls_shape)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = concatenate([x, enc_lbls_input])
        h = Dense(latent_dim * 10, activation='relu')(x)
        self.z_mean = Dense(latent_dim)(h)
        self.z_log_var = Dense(latent_dim)(h)
        z = Lambda(self.sampler, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])
        
        self.encoder = Model([encoder_input, enc_lbls_input], [self.z_mean, self.z_log_var, z])

        #self.encoder.summary()

        ########## Decoder ############

        decoder_input = Input(shape=(latent_dim,))
        dec_lbls_input = Input(shape=lbls_shape)

        x = concatenate([decoder_input, dec_lbls_input])
        x = Dense(4*4*8, activation='relu')(x)
        x = Reshape((4, 4, 8))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoder_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.decoder = Model([decoder_input, dec_lbls_input], decoder_output)

        #self.decoder.summary()

        ########## VAE #############

        vae_output = self.decoder([self.encoder([encoder_input, enc_lbls_input])[2], dec_lbls_input])

        self.vae = Model([encoder_input, enc_lbls_input, dec_lbls_input], vae_output)
        self.vae.summary()

        self.vae.compile(optimizer=optimizer, loss=self.vae_loss, metrics=['acc'])

        if(weights_path):
           self.vae.load_weights(weights_path)


    def vae_loss(self, y_true, y_pred):
        batch_size = K.shape(self.z_mean)[0]
        rec_loss = K.mean(K.binary_crossentropy(
                K.reshape(y_true, (batch_size, 784)), 
                K.reshape(y_pred, ((batch_size, 784)))
                ), axis=-1)
        kl_loss = - 0.5 * K.sum(1. + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(rec_loss + kl_loss)

    def sampler(self, args):
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # X - train data, L - labels
    def train(self, X, L, batch_size, epochs):

        checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=0,
                 save_best_only=True, save_weights_only=True,
                 mode='auto', period=1)

        return self.vae.fit([X, L, L], X, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            callbacks=[checkpoint])

    def draw_manifold(self, lbl):
        n = 15
        # Draw digits from manifold
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        figure = np.zeros((self.digit_size * n, self.digit_size * n))
        input_lbl = np.zeros((1, 10))
        input_lbl[0, lbl] = 1
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.zeros((1, self.latent_dim))
                z_sample[:, :2] = np.array([[xi, yi]])

                x_decoded = self.decoder.predict([z_sample, input_lbl])
                digit = x_decoded[0].squeeze()
                figure[i * self.digit_size: (i + 1) * self.digit_size,
                       j * self.digit_size: (j + 1) * self.digit_size] = digit

        # Visualization
        plt.figure(figsize=(10, 10), num='Manifold')
        plt.imshow(figure, cmap='Greys_r')
        plt.grid(False)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
        return figure

    def style_transfer(self, prototype, in_lbl, out_lbl):
        rows = prototype.shape[0]
        if isinstance(in_lbl, int):
            lbl = in_lbl
            in_lbl = np.zeros((rows, 10))
            in_lbl[:, lbl] = 1
        if isinstance(out_lbl, int):
            lbl = out_lbl
            out_lbl = np.zeros((rows, 10))
            out_lbl[:, lbl] = 1
        return self.vae.predict([prototype, in_lbl, out_lbl])

