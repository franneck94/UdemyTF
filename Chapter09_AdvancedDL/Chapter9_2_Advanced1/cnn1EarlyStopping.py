import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import ReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import Optimizer

from tf_utils.dogsCatsDataAdvanced import DOGSCATS


np.random.seed(0)
tf.random.set_seed(0)


LOGS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def build_model(
    optimizer: Optimizer,
    learning_rate: float,
    filter_block1: int,
    kernel_size_block1: int,
    filter_block2: int,
    kernel_size_block2: int,
    filter_block3: int,
    kernel_size_block3: int,
    dense_layer_size: int,
) -> Model:
    input_img = Input(shape=(64, 64, 3))

    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
        kernel_initializer="glorot_uniform",
    )(input_img)
    x = ReLU(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
        kernel_initializer="glorot_uniform",
    )(x)
    x = ReLU(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
        kernel_initializer="glorot_uniform",
    )(x)
    x = ReLU(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
        kernel_initializer="glorot_uniform",
    )(x)
    x = ReLU(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
        kernel_initializer="glorot_uniform",
    )(x)
    x = ReLU(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
        kernel_initializer="glorot_uniform",
    )(x)
    x = ReLU(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(
        units=dense_layer_size,
        kernel_initializer="glorot_uniform",
    )(x)
    x = ReLU(x)
    x = Dense(
        units=2,
        kernel_initializer="glorot_uniform",
    )(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred],
    )

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    epochs = 100

    data_dir = os.path.join("C:/Users/Jan/Documents/DogsAndCats")
    data = DOGSCATS(data_dir=data_dir)

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    optimizer = Adam
    learning_rate = 0.001
    filter_block1 = 32
    kernel_size_block1 = 3
    filter_block2 = 64
    kernel_size_block2 = 3
    filter_block3 = 128
    kernel_size_block3 = 7
    dense_layer_size = 512

    model = build_model(
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

    model_log_dir = os.path.join(
        LOGS_DIR,
        "model_es",
    )

    tb_callback = TensorBoard(
        log_dir=model_log_dir,
        histogram_freq=0,
        profile_batch=0,
    )

    es_callback = EarlyStopping(
        monitor="val_accuracy",
        patience=30,
        verbose=1,
        restore_best_weights=True,
        min_delta=0.0005,
    )

    model.fit(
        train_dataset,
        verbose=1,
        epochs=epochs,
        callbacks=[tb_callback, es_callback],
        validation_data=val_dataset,
    )
    scores = model.evaluate(
        val_dataset,
        verbose=0,
        batch_size=258,
    )
    print(
        f"Val performance: {scores[1]} with ES",
    )


if __name__ == "__main__":
    main()
