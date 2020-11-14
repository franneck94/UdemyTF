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


np.random.seed(0)
tf.random.set_seed(0)

LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
model_log_dir = os.path.join(LOGS_DIR, "modelCifarStart")


def build_model() -> Model:
    input_img = Input(shape=x_train.shape[1:])
    y_pred = Activation("softmax")(input_img)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    cifar = CIFAR10()
    cifar.data_augmentation(augment_size=5_000)
    cifar.data_preprocessing(preprocess_mode="MinMax")
    x_train, y_train = cifar.get_train_set()
    x_test, y_test = cifar.get_test_set()
    num_classes = cifar.num_classes
