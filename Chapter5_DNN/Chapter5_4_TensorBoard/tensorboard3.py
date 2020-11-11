import os


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from utils.plotting import ConfusionMatrix


MODEL_DIR = os.path.abspath("C:/Users/jan/Dropbox/_Programmieren/UdemyTF/models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "mnist_model.h5")
LOG_DIR = os.path.abspath("C:/Users/jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
MODEL_LOG_DIR = os.path.join(LOG_DIR, "mnist_cm")


def prepare_dataset(num_features: int, num_classes: int):
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
    model.add(Dense(units=500, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,),))
    model.add(Activation("relu"))
    model.add(Dense(units=300, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=50, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_features, num_classes)

    optimizer = Adam(learning_rate=0.001)
    epochs = 1
    batch_size = 256

    model = build_model(num_features, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        histogram_freq=1,
        write_graph=True
    )

    classes_list = [class_idx for class_idx in range(num_classes)]

    cm_callback = ConfusionMatrix(
        model,
        x_test,
        y_test,
        classes_list=classes_list,
        log_dir=MODEL_LOG_DIR
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback, cm_callback],
    )

    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0
    )
    print("Scores: ", scores)
