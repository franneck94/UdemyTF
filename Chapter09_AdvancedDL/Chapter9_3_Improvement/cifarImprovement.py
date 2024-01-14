import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.initializers import Initializer
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import ReLU
from keras.models import Model
from keras.optimizers import Optimizer
from keras.optimizers import RMSprop

from tf_utils.callbacks import LRTensorBoard
from tf_utils.callbacks import schedule_fn
from tf_utils.cifarDataAdvanced import CIFAR10


np.random.seed(0)  # noqa: NPY002
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
    filter_block4: int,
    kernel_size_block4: int,
    dense_layer_size: int,
    kernel_initializer: Initializer,
    activation_cls: Activation,
    dropout_rate: float,
    use_batch_normalization: bool,
    use_dense: bool,
    use_global_pooling: bool,
) -> Model:
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(input_img)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block4,
        kernel_size=kernel_size_block4,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block4,
        kernel_size=kernel_size_block4,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    if use_global_pooling:
        x = GlobalAveragePooling2D()(x)
    else:
        Flatten()(x)
    if use_dense:
        x = Dense(
            units=dense_layer_size,
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = activation_cls(x)
    x = Dense(
        units=10,
        kernel_initializer=kernel_initializer,
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
    model.summary()

    return model


if __name__ == "__main__":
    epochs = 100

    data = CIFAR10()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    params = {
        "dense_layer_size": 64,
        "kernel_initializer": "GlorotUniform",
        "optimizer": RMSprop,
        "learning_rate": 5e-3,
        "filter_block1": 32,
        "kernel_size_block1": 3,
        "filter_block2": 64,
        "kernel_size_block2": 3,
        "filter_block3": 128,
        "kernel_size_block3": 5,
        "filter_block4": 128,
        "kernel_size_block4": 5,
        "activation_cls": ReLU(),
        "dropout_rate": 0.0,
        "use_batch_normalization": True,
        "use_dense": True,
        "use_global_pooling": True,
    }

    model = build_model(
        **params,
    )

    model_log_dir = os.path.join(LOGS_DIR, "model_Final_CIFAR5")

    lr_callback = LRTensorBoard(
        log_dir=model_log_dir,
        histogram_freq=0,
        profile_batch=0,
    )

    lrs_callback = LearningRateScheduler(
        schedule=schedule_fn,
        verbose=1,
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
        callbacks=[lr_callback, lrs_callback, es_callback],
        validation_data=val_dataset,
    )

    scores = model.evaluate(
        val_dataset,
        verbose=0,
        batch_size=256,
    )
    print(f"Scores: {scores}")
