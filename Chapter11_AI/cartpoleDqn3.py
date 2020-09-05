import os
import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

class DQN(Model):
    def __init__(self, state_shape, num_actions, lr):
        super(DQN, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr

        input_state = Input(shape=state_shape)
        x = Dense(20)(input_state)
        x = Activation("relu")(x)
        x = Dense(20)(x)
        x = Activation("relu")(x)
        output_pred = Dense(self.num_actions)(x)

        self.model = Model(inputs=input_state, outputs=output_pred)
        self.model.compile(loss="mse", optimizer=Adam(lr=self.lr))

    def train(self, states, q_values):
        self.model.fit(states, q_values, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)