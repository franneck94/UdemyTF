from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import boston_housing


np.random.seed(0)
tf.random.set_seed(0)


class BOSTON:
    def __init__(self, with_normalization: bool = True) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        self.x_train_: np.ndarray = None
        self.y_train_: np.ndarray = None
        self.x_val_: np.ndarray = None
        self.y_val_: np.ndarray = None
        self.val_size = 0
        self.train_splitted_size = 0
        # Preprocess x data
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.num_targets = 1
        # Preprocess y data
        self.y_train = np.reshape(self.y_train, (-1, 1))
        self.y_test = np.reshape(self.y_test, (-1, 1))

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size: float = 0.33) -> tuple:
        self.x_train_, self.x_val_, self.y_train_, self.y_val_ = train_test_split(
            self.x_train,
            self.y_train,
            test_size=validation_size
        )
        self.val_size = self.x_val_.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val_, self.y_train_, self.y_val_


if __name__ == "__main__":
    data = BOSTON()
