import os
from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/computation/")
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "gradient_model")


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[i, i] for i in range(100)],
        dtype=np.float32
    )
    y = np.array(
        [i for i in range(100)],
        dtype=np.float32
    ).reshape(-1, 1)
    return x, y


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(1, input_shape=(2,), name="hidden"))
    model.add(Activation("relu", name="relu"))
    model.add(Dense(1, name="output"))
    model.summary()
    return model


def get_gradients(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: Sequential,
    loss_object: tf.keras.losses.Loss
) -> List[Tuple[np.ndarray, np.ndarray]]:
    with tf.GradientTape() as tape:
        y_pred = model(x_test, training=True)
        loss_value = loss_object(y_test, y_pred)
    grads = tape.gradient(loss_value, model.trainable_variables)
    grad_var_tuples = [
        (g, w) for (g, w) in zip(grads, model.trainable_variables)
    ]
    return grad_var_tuples


if __name__ == "__main__":
    x, y = get_dataset()

    model = build_model()

    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=1e-2),
        metrics=["mse"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        embeddings_freq=0,
        write_graph=True
    )

    model.fit(
        x=x,
        y=y,
        verbose=1,
        batch_size=1,
        epochs=0,
        callbacks=[tb_callback]
    )

    model.layers[0].set_weights(
        [np.array([[-0.250], [1.000]]),
         np.array([0.100])]
    )
    model.layers[2].set_weights(
        [np.array([[1.250]]),
         np.array([0.125])]
    )

    ###############
    ### TESTING ###
    ###############
    loss_object = MeanSquaredError()

    x_test = np.array([[2, 2]])
    y_test = np.array([[2]])

    y_pred = model.predict(x_test)
    print(f"Pred: {y_pred}")

    layer_names = [
        "hidden:kernel"
        "hidden:bias",
        "output:kernel",
        "output:bias"
    ]
    gradients = get_gradients(x_test, y_test, model, loss_object)

    for name, (grads, weight) in zip(layer_names, gradients):
        print(f"Name:\n{name}")
        print(f"Weights:\n{weight.numpy()}")
        print(f"Grads:\n{grads.numpy()}\n")
