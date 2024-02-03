import os

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from tensorcross.model_selection import GridSearch

from tf_utils.cifarDataAdvanced import CIFAR10


np.random.seed(0)
tf.random.set_seed(0)

LOGS_DIR = os.path.abspath(
    "C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs/cifarGrid",
)
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def build_model(
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
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
    )(input_img)
    x = Activation("relu")(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
    )(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
    )(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
    )(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
    )(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
    )(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=dense_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(units=10)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    epochs = 30

    data = CIFAR10(augment=True, shuffle=True)

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    param_grid = {
        "optimizer": [Adam, RMSprop],
        "learning_rate": [0.001, 0.002],
        "filter_block1": [32],
        "kernel_size_block1": [3, 5],
        "filter_block2": [32, 64],
        "kernel_size_block2": [3, 5],
        "filter_block3": [64, 128],
        "kernel_size_block3": [7],
        "dense_layer_size": [512],
    }

    tb_callback = TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=0,
        profile_batch=0,
    )

    grid_search = GridSearch(
        model_fn=build_model,
        param_grid=param_grid,
        verbose=1,
    )

    grid_search.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[tb_callback],
    )

    grid_search.summary()


if __name__ == "__main__":
    main()
