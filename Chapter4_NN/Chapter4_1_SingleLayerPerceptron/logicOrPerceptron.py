from typing import Tuple

import numpy as np


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """OR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    return x, y


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pass


def step_function(input_signal: np.ndarray) -> np.ndarray:
    pass


class Perceptron:
    def __init__(self) -> None:
        pass

    def _update_weights(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        pass

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        pass


if __name__ == "__main__":
    x, y = get_dataset()

    p = Perceptron()
