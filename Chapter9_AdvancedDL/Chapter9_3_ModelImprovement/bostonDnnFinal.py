import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tf_utils.bostonDataAdvanced import BOSTON
from tf_utils.callbacks import LRTensorBoard
from tf_utils.callbacks import schedule_fn5


np.random.seed(0)
tf.random.set_seed(0)


LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    error = tf.math.subtract(y_true, y_pred)
    squared_error = tf.math.square(error)
    numerator = tf.math.reduce_sum(squared_error)
    y_true_mean = tf.math.reduce_mean(y_true)
    mean_deviation = tf.math.subtract(y_true, y_true_mean)
    squared_mean_deviation = tf.math.square(mean_deviation)
    denominator = tf.reduce_sum(squared_mean_deviation)
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


def build_model(
    num_features: int,
    num_targets: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float,
    dense_layer_size1: int,
    dense_layer_size2: int,
    activation_str: str,
    dropout_rate: bool,
    use_batch_normalization: bool,
) -> Model:
    # Input
    input_house = Input(shape=num_features)
    # Dense Layer 1
    x = Dense(units=dense_layer_size1)(input_house)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    # Dense Layer 2
    x = Dense(units=dense_layer_size2)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    # Output Layer
    x = Dense(units=num_targets)(x)
    y_pred = Activation("linear")(x)

    model = Model(
        inputs=[input_house],
        outputs=[y_pred]
    )

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="mse",
        optimizer=opt,
        metrics=[r_squared]
    )
    model.summary()

    return model


if __name__ == "__main__":
    """
    Chapter 5: 0.7287 R2
    LinReg: 0.7174 R2
    Model: 0.8197 R2
    """
    data = BOSTON()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    num_features = data.num_features
    num_targets = data.num_targets

    # Global params
    epochs = 2_000
    batch_size = 64

    params = {
        "optimizer": Adam,
        "learning_rate": 0.001,
        "dense_layer_size1": 256,
        "dense_layer_size2": 128,
        # relu, elu, LeakyReLU
        "activation_str": "relu",
        # 0.05, 0.1, 0.2
        "dropout_rate": 0.00,
        # True, False
        "use_batch_normalization": True
    }

    model = build_model(
        num_features,
        num_targets,
        **params
    )

    model_log_dir = os.path.join(LOGS_DIR, "model_Final_BOSTON")

    lr_callback = LRTensorBoard(
        log_dir=model_log_dir,
        histogram_freq=0,
        profile_batch=0,
        write_graph=False
    )

    lrs_callback = LearningRateScheduler(
        schedule=schedule_fn5,
        verbose=0
    )

    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
        verbose=2,
        restore_best_weights=True,
        min_delta=0.0005
    )

    model.fit(
        train_dataset,
        verbose=1,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[lr_callback, lrs_callback, es_callback],
        validation_data=val_dataset
    )

    score = model.evaluate(
        test_dataset,
        verbose=0,
        batch_size=batch_size
    )
    print(f"Test performance: {score}")
