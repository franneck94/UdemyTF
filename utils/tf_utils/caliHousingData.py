from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


np.random.seed(0)


class CALIHOUSING:
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.33) -> None:
        # User-definen constants
        self.num_targets = 1
        # Load the data set
        dataset = fetch_california_housing()
        self.x, self.y = dataset.data, dataset.target
        self.feature_names = dataset.feature_names
        self.description = dataset.DESCR
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y data
        self.y_train = np.reshape(y_train, (-1, self.num_targets)).astype(np.float32)
        self.y_test = np.reshape(y_test, (-1, self.num_targets)).astype(np.float32)
        self.y_val = np.reshape(y_val, (-1, self.num_targets)).astype(np.float32)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.num_features = self.x_train.shape[1]
        self.num_targets = self.y_train.shape[1]

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_val, self.y_val
