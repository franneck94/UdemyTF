import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical


LOGS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_cnn4")


def prepare_dataset(num_classes: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = x_test.astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(filters=32, kernel_size=3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    model.summary()

    return model


def plot_filters(model: Model) -> None:
    first_conv_layer = model.layers[1]
    layer_weights = first_conv_layer.get_weights()
    kernels = layer_weights[0]

    num_filters = kernels.shape[3]
    subplot_grid = (num_filters // 4, 4)

    fig, ax = plt.subplots(subplot_grid[0], subplot_grid[1], figsize=(20, 20))
    ax = ax.reshape(num_filters)

    for filter_idx in range(num_filters):
        ax[filter_idx].imshow(kernels[:, :, 0, filter_idx], cmap="gray")

    ax = ax.reshape(subplot_grid)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_classes)

    model = build_model(img_shape, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR, histogram_freq=1, write_graph=True
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback],
    )

    scores = model.evaluate(x=x_test, y=y_test, verbose=0)
    print(scores)

    plot_filters(model)
