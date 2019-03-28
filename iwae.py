import tensorflow.examples.tutorials.mnist
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
import keras.optimizers as optimizers
from keras.models import load_model
import pickle

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras import losses
import sys

m = 20
n_z = 50
epochs = 100
image_dim = 28
input_shape = image_dim * image_dim
deterministic_shape = 200
k = 5
log2pi = K.log(2 * np.pi)

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

inputs = Input(shape=(input_shape,), name='encoder_input')
encoder_first_deterministic_layer = Dense(deterministic_shape, activation='relu')(inputs)
encoder_second_deterministic_layer = Dense(deterministic_shape, activation='relu')(encoder_first_deterministic_layer)
mu = Dense(n_z, activation='linear')(encoder_second_deterministic_layer)
sigma = Dense(n_z, activation='softplus')(encoder_second_deterministic_layer)

def sample_z(args):
    k = 5
    local_mu, local_sigma = args
    local_mu = K.repeat(local_mu, k)
    local_sigma = K.repeat(local_sigma, k)
    eps = K.random_normal(shape=(K.shape(local_mu)[0], k, K.shape(local_mu)[2]), mean=0., stddev=1.)
    return local_mu + local_sigma * eps

z = Lambda(sample_z, output_shape=(k, n_z,), name='z')([mu, sigma])
decoder_first_deterministic_layer = Dense(deterministic_shape, activation='relu')
decoder_second_deterministic_layer = Dense(deterministic_shape, activation='relu')
decoder_out = Dense(input_shape, activation='sigmoid')

vae_first_deterministic_layer = decoder_first_deterministic_layer(z)
vae_second_deterministic_layer = decoder_second_deterministic_layer(vae_first_deterministic_layer)
vae_outputs = decoder_out(vae_second_deterministic_layer)
vae = Model(inputs, vae_outputs)

encoder = Model(inputs, [mu, sigma, z], name='encoder')

generator_input = Input(shape=(n_z,), name='z_sampling')
generator_first_deterministic_layer = decoder_first_deterministic_layer(generator_input)
generator_second_deterministic_layer = decoder_second_deterministic_layer(generator_first_deterministic_layer)
generator_output = decoder_out(generator_second_deterministic_layer)
decoder = Model(generator_input, generator_output, name='decoder')

def iwae_loss(y_true, y_pred):
    local_mu = K.repeat(mu, k)
    local_sigma = K.repeat(sigma, k)
    log_posterior = -(n_z / 2) * log2pi - K.sum(K.log(1e-8 + local_sigma) + 0.5 * K.square(z - local_mu) / K.square(1e-8 + local_sigma), axis=-1)
    log_prior = -(n_z / 2) * log2pi - K.sum(0.5 * K.square(z), axis=-1)
    log_bernoulli = K.sum(y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8), axis=-1)
    log_weights = log_bernoulli + log_prior - log_posterior
    importance_weight = K.softmax(log_weights, axis=1)
    return -K.sum(importance_weight * log_weights, axis=-1)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4)
vae.compile(optimizer=adam, loss=iwae_loss)
history = vae.fit(x_train, np.rollaxis(np.tile(x_train, reps=(k, 1, 1)), axis=1), shuffle=True, batch_size=m, epochs=epochs, validation_data=(x_test, np.rollaxis(np.tile(x_test, reps=(k, 1, 1)), axis=1)))

vae.save("mnist-iwae.h5")
encoder.save("mnist-encoder-iwae.h5")
decoder.save("mnist-decoder-iwae.h5")
with open("mnist-history-iwae.pickle", "wb") as file_pi:
    pickle.dump(history.history, file_pi)