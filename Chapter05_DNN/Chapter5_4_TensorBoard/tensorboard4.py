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

from tf_utils.callbacks import ConfusionMatrix


MODEL_DIR = os.path.abspath("C:/Users/jan/OneDrive/_Coding/UdemyTF/models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "mnist_model.h5")
LOGS_DIR = os.path.abspath("C:/Users/jan/OneDrive/_Coding/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_cm")


def prepare_dataset(num_features: int, num_classes: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_classes: int) -> Sequential:
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
        Dense(units=300, kernel_initializer=init_w, bias_initializer=init_b)
    )
    model.add(Activation("relu"))
    model.add(
        Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b)
    )
    model.add(Activation("relu"))
    model.add(
        Dense(units=50, kernel_initializer=init_w, bias_initializer=init_b)
    )
    model.add(Activation("relu"))
    model.add(
        Dense(
            units=num_classes,
            kernel_initializer=init_w,
            bias_initializer=init_b,
        )
    )
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(
        num_features, num_classes
    )

    optimizer = Adam(learning_rate=0.001)
    epochs = 5
    batch_size = 256

    model = build_model(num_features, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR, histogram_freq=1, write_graph=True
    )

    classes_list = list(range(num_classes))

    cm_callback = ConfusionMatrix(
        model, x_test, y_test, classes_list=classes_list, log_dir=MODEL_LOG_DIR
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback, cm_callback],
    )

    scores = model.evaluate(x=x_test, y=y_test, verbose=0)
    print("Scores: ", scores)
