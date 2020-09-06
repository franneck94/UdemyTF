import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def sigmoid_fn(x):
    pass


def tanh_fn(x):
    pass


class SimpleRNNInference:
    def __init__(self):
        pass

    def __call__(self):
        pass

    def forward_step(self):
        pass


# data set shape = (num_samples, num_timesteps, num_features)
# input shape = (num_timesteps, num_features)
# If return_sequences == True:
# output shape = (num_timesteps, units)
# Else:
# output shape = (1, units)
x = np.random.normal(size=(1, 3, 2))
units = 4
return_sequences = False

model = Sequential()
model.add(SimpleRNN(units=units, return_sequences=return_sequences, input_shape=x.shape[1:]))
model.compile(loss="mse", optimizer="Adam")
model.summary()
