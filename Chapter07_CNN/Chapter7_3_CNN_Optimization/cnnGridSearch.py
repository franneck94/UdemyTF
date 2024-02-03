import numpy as np
import tensorflow as tf
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.models import Model
from tensorcross.model_selection import GridSearch

from tf_utils.mnistDataAdvanced import MNIST


np.random.seed(0)
tf.random.set_seed(0)


def build_model(
    filters_1: int,
    kernel_size_1: int,
    filters_2: int,
    kernel_size_2: int,
    filters_3: int,
    kernel_size_3: int,
) -> Model:
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(filters=filters_1, kernel_size=kernel_size_1, padding="same")(
        input_img,
    )
    x = Activation("relu")(x)
    x = Conv2D(filters=filters_1, kernel_size=kernel_size_1, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=filters_2, kernel_size=kernel_size_2, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filters_2, kernel_size=kernel_size_2, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=filters_3, kernel_size=kernel_size_3, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filters_3, kernel_size=kernel_size_3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=10)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred],
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam",
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    data = MNIST(augment=True, shuffle=True)

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    param_grid = {
        "filters_1": [16, 32],
        "kernel_size_1": [3, 5],
        "filters_2": [32, 64],
        "kernel_size_2": [3, 5],
        "filters_3": [64, 128],
        "kernel_size_3": [5],
    }

    grid_search = GridSearch(
        model_fn=build_model,
        param_grid=param_grid,
        verbose=1,
    )

    grid_search.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,
        verbose=1,
    )

    grid_search.summary()


if __name__ == "__main__":
    main()
