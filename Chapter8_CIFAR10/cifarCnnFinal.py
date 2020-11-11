import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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


def model_fn(
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float,
    filter_block1: int,
    kernel_size_block1: int,
    filter_block2: int,
    kernel_size_block2: int,
    filter_block3: int,
    kernel_size_block3: int,
    dense_layer_size: int,
) -> Model:
    # Input
    input_img = Input(shape=x_train.shape[1:])
    # Conv Block 1
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same')(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Conv Block 2
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Conv Block 3
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Dense Part
    x = Flatten()(x)
    x = Dense(units=dense_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    # Build the model
    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )
    opt = optimizer(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    return model


# Global params
epochs = 50
batch_size = 256

# Best grid search model
optimizer = Adam
learning_rate = 1e-3
filter_block1 = 32
kernel_size_block1 = 3
filter_block2 = 64
kernel_size_block2 = 5
filter_block3 = 7
kernel_size_block3 = 64
dense_layer_size = 512

grid_model = model_fn(
    optimizer,
    learning_rate,
    filter_block1,
    kernel_size_block1,
    filter_block2,
    kernel_size_block2,
    filter_block3,
    kernel_size_block3,
    dense_layer_size,
)
model_log_dir = os.path.join(log_dir, "modelBestGrid")

tb_callback = TensorBoard(
    log_dir=model_log_dir
)

grid_model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tb_callback],
    validation_data=(x_test, y_test),
)
score = grid_model.evaluate(
    x_test,
    y_test,
    verbose=0,
    batch_size=batch_size
)
print("Test performance best grid model: ", score)

# Best random model
optimizer = Adam
learning_rate = 0.0006214855772522395
filter_block1 = 43
kernel_size_block1 = 3
filter_block2 = 50
kernel_size_block2 = 3
filter_block3 = 54
kernel_size_block3 = 4
dense_layer_size = 844

rand_model = model_fn(
    optimizer,
    learning_rate,
    filter_block1,
    kernel_size_block1,
    filter_block2,
    kernel_size_block2,
    filter_block3,
    kernel_size_block3,
    dense_layer_size,
)
model_log_dir = os.path.join(log_dir, "modelBestRand")

tb_callback = TensorBoard(
    log_dir=model_log_dir
)

rand_model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tb_callback],
    validation_data=(x_test, y_test),
)
score = rand_model.evaluate(
    x_test,
    y_test,
    verbose=0,
    batch_size=batch_size
)
print("Test performance best rand model: ", score)

# Huge model
optimizer = Adam
learning_rate = 0.0005
filter_block1 = 32
kernel_size_block1 = 35
filter_block2 = 64
kernel_size_block2 = 4
filter_block3 = 128
kernel_size_block3 = 3
dense_layer_size = 2048

rand_model = model_fn(
    optimizer,
    learning_rate,
    filter_block1,
    kernel_size_block1,
    filter_block2,
    kernel_size_block2,
    filter_block3,
    kernel_size_block3,
    dense_layer_size,
)
model_log_dir = os.path.join(log_dir, "modelHuge")

tb_callback = TensorBoard(
    log_dir=model_log_dir
)

rand_model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tb_callback],
    validation_data=(x_test, y_test),
)
score = rand_model.evaluate(
    x_test,
    y_test,
    verbose=0,
    batch_size=batch_size
)
print("Test performance best rand model: ", score)
