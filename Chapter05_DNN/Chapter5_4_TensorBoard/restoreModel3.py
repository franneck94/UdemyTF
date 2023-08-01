import os

import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical


MODELS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/models")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "full_mnist_model.h5")


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


if __name__ == "__main__":
    num_features = 784
    num_targets = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(
        num_features, num_targets
    )

    model = load_model(MODEL_FILE_PATH)

    scores = model.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Scores after loading: {scores}")
