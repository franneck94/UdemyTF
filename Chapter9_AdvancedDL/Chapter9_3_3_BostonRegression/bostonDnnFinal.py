import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from bostonData import BOSTON
from tensorflow_addons.metrics import r_square


np.random.seed(0)
tf.random.set_seed(0)

LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def build_model(
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float,
    dense_layer_size1: int,
    dense_layer_size2: int,
    activation_str: str,
    dropout_rate: bool,
    use_bn: bool,
) -> Model:
    # Input
    input_house = Input(shape=x_train.shape[1:])
    # Dense Layer 1
    x = Dense(units=dense_layer_size1)(input_house)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    # Dense Layer 2
    x = Dense(units=dense_layer_size2)(x)
    if use_bn:
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

    # Build the model
    model = Model(
        inputs=[input_house],
        outputs=[y_pred]
    )
    opt = optimizer(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=opt, metrics=[r_square])
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
    threshold = 500
    if epoch < threshold:
        return 1e-3
    else:
        return 1e-3 * np.exp(0.005 * (threshold - epoch))


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir: str, **kwargs: dict) -> None:
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        logs.update({'learning_rate': self.model.optimizer.learning_rate})
        super().on_epoch_end(epoch, logs)


if __name__ == "__main__":

    data = BOSTON()

    (x_train_, x_val, y_train_, y_val,) = data.get_splitted_train_validation_set()
    x_train, y_train = data.get_train_set()
    x_test, y_test = data.get_test_set()

    num_targets = data.num_targets

    # Global params
    epochs = 2000
    batch_size = 256

    params = {
        "optimizer": Adam,
        "learning_rate": 0.001,
        "dense_layer_size1": 128,
        "dense_layer_size2": 64,
        # relu, elu, LeakyReLU
        "activation_str": "relu",
        # 0.05, 0.1, 0.2
        "dropout_rate": 0.00,
        # True, False
        "use_bn": True,
    }

    rand_model = build_model(**params)

    # Model 1: schedule_fn1
    # Model 2: schedule_fn2
    lrs_callback = LearningRateScheduler(
        schedule=schedule_fn2,
        verbose=1
    )

    # Model 3: factor=0.95
    plateau_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.98,
        patience=50,
        verbose=1,
        min_lr=1e-5
    )

    es_callback = EarlyStopping(
        monitor='val_loss',
        patience=200,
        verbose=1,
        restore_best_weights=True
    )

    model_log_dir = os.path.join(LOGS_DIR, "modelBostonFinal6")
    tb_callback = LRTensorBoard(log_dir=model_log_dir)

    rand_model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tb_callback, lrs_callback, es_callback],
        validation_data=(x_test, y_test),
    )

    score = rand_model.evaluate(
        x_test,
        y_test,
        verbose=0,
        batch_size=batch_size
    )
    print(f"Test performance: {score}")
