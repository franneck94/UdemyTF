import os

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.optimizers import Adam


LOGS_DIR = os.path.abspath(
    "C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs/computation/"
)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "gradient_model")


def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    num_samples = 100
    x = np.array(
        [[i, i] for i in range(num_samples)],
        dtype=np.float32,
    )
    y = np.array(
        list(range(num_samples)),
        dtype=np.float32,
    ).reshape(-1, 1)
    return x, y


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(units=1, input_shape=(2,), name="hidden"))
    model.add(Activation("relu", name="relu"))
    model.add(Dense(units=1, name="output"))
    model.summary()
    return model


def get_gradients(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: Sequential,
    loss_object: tf.keras.losses.Loss,
) -> list[tuple[tf.Tensor, tf.Tensor]]:
    with tf.GradientTape() as tape:
        y_pred = model(x_test, training=True)
        loss_value = loss_object(y_test, y_pred)
    grads = tape.gradient(
        loss_value,
        model.trainable_variables,
    )
    return [(g, w) for (g, w) in zip(grads, model.trainable_variables)]


def main() -> None:
    x, y = get_dataset()

    model = build_model()

    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=1e-2),
        metrics=["mse"],
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        embeddings_freq=0,
        write_graph=True,
    )

    model.fit(
        x=x,
        y=y,
        verbose=1,
        batch_size=1,
        epochs=0,
        callbacks=[tb_callback],
    )

    model.layers[0].set_weights(
        [
            np.array([[-0.250], [1.000]]),
            np.array([0.100]),
        ]
    )
    model.layers[2].set_weights(
        [
            np.array([[1.250]]),
            np.array([0.125]),
        ]
    )

    # Test
    loss_object = MeanSquaredError()

    x_test = np.array([[2, 2]])
    y_test = np.array([[2]])

    y_pred = model.predict(x_test)
    print(f"Pred: {y_pred}")

    gradients = get_gradients(
        x_test,
        y_test,
        model,
        loss_object,
    )

    for grads, weight in gradients:
        print(f"Layer name: {weight.name}")
        print(f"Weights:\n{weight.numpy()}")
        print(f"Grads:\n{grads.numpy()}\n")


if __name__ == "__main__":
    main()
