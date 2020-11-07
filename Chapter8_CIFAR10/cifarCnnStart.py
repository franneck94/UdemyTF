import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

from cifar10Data import CIFAR10


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

cifar = CIFAR10()
cifar.data_augmentation(augment_size=5000)
cifar.data_preprocessing(preprocess_mode="MinMax")
x_train, y_train = cifar.get_train_set()
x_test, y_test = cifar.get_test_set()
num_classes = cifar.num_classes

# Save Path
dir_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/models/")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
cifar_model_path = os.path.join(dir_path, "cifar_model.h5")
# Log Dir
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
model_log_dir = os.path.join(log_dir, "modelCifarStart")


def model_fn():
    pass
    # # Input
    # input_img = Input(shape=x_train.shape[1:])
    # # ...
    # # Output
    # x = Dense(units=num_classes)(x)
    # y_pred = Activation("softmax")(x)

    # # Build the model
    # model = Model(
    #     inputs=[input_img],
    #     outputs=[y_pred]
    # )
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # return model
