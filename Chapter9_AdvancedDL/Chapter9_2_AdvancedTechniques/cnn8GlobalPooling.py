import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from dogsCatsData import DOGSCATS


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


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
    kernel_initializer: tf.keras.initializers.Initializer,
    bias_initializer: tf.keras.initializers.Initializer,
    activation_str: str,
    dropout_rate: float,
    use_bn: bool,
    use_global_pooling: bool,
    use_additional_dense_layer: bool
) -> Model:
    # Input
    input_img = Input(shape=x_train.shape[1:])
    # Conv Block 1
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(input_img)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 2
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Dense Part
    if use_global_pooling:
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    if use_additional_dense_layer:
        x = Dense(units=dense_layer_size)(x)
        if activation_str == "LeakyReLU":
            x = LeakyReLU()(x)
        else:
            x = Activation(activation_str)(x)
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
    model.summary()
    return model


def schedule_fn(epoch: int) -> float:
    learning_rate = 1e-3
    if epoch < 5:
        learning_rate = 1e-3
    elif epoch < 20:
        learning_rate = 5e-4
    else:
        learning_rate = 1e-4
    return learning_rate


def schedule_fn2(epoch: int) -> float:
    if epoch < 10:
        return 1e-3
    else:
        return 1e-3 * np.exp(0.1 * (10 - epoch))


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir: str, **kwargs: dict) -> None:
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        logs.update({'learning_rate': self.model.optimizer.learning_rate})
        super().on_epoch_end(epoch, logs)


if __name__ == "__main__":
    data = DOGSCATS()
    data.data_augmentation(augment_size=5_000)
    data.data_preprocessing(preprocess_mode="MinMax")
    (x_train_splitted, x_val, y_train_splitted, y_val,) = data.get_splitted_train_validation_set()
    x_train, y_train = data.get_train_set()
    x_test, y_test = data.get_test_set()
    num_classes = data.num_classes

    # Global params
    epochs = 100
    batch_size = 256

    params = {
        "optimizer": Adam,
        "learning_rate": 0.001,
        "filter_block1": 32,
        "kernel_size_block1": 3,
        "filter_block2": 64,
        "kernel_size_block2": 3,
        "filter_block3": 128,
        "kernel_size_block3": 3,
        "dense_layer_size": 1024,
        # GlorotUniform, GlorotNormal, RandomNormal
        # RandomUniform, VarianceScaling
        "kernel_initializer": 'GlorotUniform',
        "bias_initializer": 'zeros',
        # relu, elu, LeakyReLU
        "activation_str": "relu",
        # 0.05, 0.1, 0.2
        "dropout_rate": 0.00,
        # True, False
        "use_bn": True,
        # True, False
        "use_global_pooling": True,
        # True, False
        "use_additional_dense_layer": True,
    }

    rand_model = model_fn(**params)

    lrs_callback = LearningRateScheduler(
        schedule=schedule_fn2,
        verbose=1
    )

    plateau_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.90,
        patience=2,
        verbose=1,
        min_lr=1e-5
    )

    es_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    model_log_dir = os.path.join(LOGS_DIR, "modelGlobalDense")
    tb_callback = LRTensorBoard(log_dir=model_log_dir)

    rand_model.fit(
        x=x_train_splitted,
        y=y_train_splitted,
        verbose=1,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tb_callback, plateau_callback, es_callback],
        validation_data=(x_val, y_val),
    )

    score = rand_model.evaluate(
        x_test,
        y_test,
        verbose=0,
        batch_size=batch_size
    )
    print(f"Test performance: {score}")
