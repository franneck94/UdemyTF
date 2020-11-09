from typing import Tuple

import numpy as np
import tensorflow as tf


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """XOR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return x, y


class Model:
    def __init__(self, optimizer, loss, metric) -> None:
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric

    def _update_weights(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            continue

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        pass


if __name__ == "__main__":
    x, y = get_dataset()

    num_features = 2
    num_targets = 1

    learning_rate = 0.5
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsoluteError()
    metric = tf.keras.metrics.BinaryAccuracy()

    model = Model(optimizer, loss, metric)
    model.fit(x, y, epochs=10)
