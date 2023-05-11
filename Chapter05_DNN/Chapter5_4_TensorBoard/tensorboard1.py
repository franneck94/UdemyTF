import os

import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.initializers import Constant
from keras.initializers import TruncatedNormal
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


LOGS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_model")


def prepare_dataset(num_features: int, num_targets: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_targets, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_targets, dtype=np.float32)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_targets: int) -> Sequential:
    init_w = TruncatedNormal(mean=0.0, stddev=0.01)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(
        Dense(
            units=500,
            kernel_initializer=init_w,
            bias_initializer=init_b,
            input_shape=(num_features,),
        )
    )
    model.add(Activation("relu"))
    model.add(
        Dense(units=250, kernel_initializer=init_w, bias_initializer=init_b)
    )
    model.add(Activation("relu"))
    model.add(
        Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b)
    )
    model.add(Activation("relu"))
    model.add(
        Dense(
            units=num_targets,
            kernel_initializer=init_w,
            bias_initializer=init_b,
        )
    )
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":
    num_features = 784
    num_targets = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(
        num_features, num_targets
    )

    model = build_model(num_features, num_targets)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR, histogram_freq=1, write_graph=True
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback],
    )

    scores = model.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Scores before saving: {scores}")
