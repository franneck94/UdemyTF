import os
from typing import Tuple

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from mnistData import MNIST


LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "cnn_no_norm_augment")


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(filters=32, kernel_size=3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )

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
    data = MNIST(with_normalization=False)

    x_train, y_train = data.get_train_set()
    x_test, y_test = data.get_test_set()

    model = build_model(data.img_shape, data.num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        write_graph=True
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=40,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback]
    )

    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0
    )
    print(scores)
