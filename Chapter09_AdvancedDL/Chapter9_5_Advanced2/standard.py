import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from tf_utils.dogsCatsDataAdvanced import DOGSCATS
from tf_utils.mnistDataAdvanced import MNIST  # noqa


np.random.seed(0)
tf.random.set_seed(0)


LOGS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def relu_norm(x: tf.Tensor) -> tf.Tensor:
    x = Activation("relu")(x)
    return BatchNormalization()(x)


def conv_block(
    x: tf.Tensor,
    filters: int,
    downsample: bool = False,
) -> tf.Tensor:
    x = Conv2D(
        filters=filters,
        strides=(1 if not downsample else 2),
        kernel_size=3,
        padding="same",
    )(x)
    x = relu_norm(x)
    x = Conv2D(
        filters=filters,
        strides=1,
        kernel_size=3,
        padding="same",
    )(x)
    return relu_norm(x)


def output_block(
    x: tf.Tensor,
    num_classes: int,
) -> tf.Tensor:
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes)(x)
    return Activation("softmax")(x)


def build_model_standard(
    img_shape: tuple[int, int, int],
    num_classes: int,
) -> Model:
    input_img = Input(shape=img_shape)

    x = conv_block(
        x=input_img,
        filters=32,
        downsample=False,
    )
    x = conv_block(
        x=x,
        filters=64,
        downsample=True,
    )
    x = conv_block(
        x=x,
        filters=64,
        downsample=False,
    )
    x = conv_block(
        x=x,
        filters=128,
        downsample=True,
    )
    x = conv_block(
        x=x,
        filters=128,
        downsample=False,
    )
    y_pred = output_block(
        x=x,
        num_classes=num_classes,
    )

    model = Model(
        inputs=[input_img],
        outputs=[y_pred],
    )

    opt = Adam()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    epochs = 100

    # data = MNIST()
    data_dir = os.path.join("C:/Users/Jan/Documents/DogsAndCats")
    data = DOGSCATS(data_dir=data_dir)

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    model = build_model_standard(img_shape, num_classes)

    model.summary()

    model_log_dir = os.path.join(
        LOGS_DIR,
        f"model_standard_{data.__class__.__name__}",
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
        callbacks=[es_callback],
        validation_data=val_dataset,
    )

    scores = model.evaluate(
        val_dataset,
        verbose=0,
        batch_size=256,
    )
    print(f"Scores: {scores}")
