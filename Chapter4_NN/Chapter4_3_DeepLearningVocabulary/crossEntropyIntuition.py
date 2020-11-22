from typing import Tuple

import numpy as np


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """OR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    return x, y


def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    y_categorical = np.zeros(shape=(len(y), num_classes))
    for i, yi in enumerate(y):
        y_categorical[i, yi] = 1
    return y_categorical


def softmax(y_pred: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        exps = np.exp(y_pred[i])
        probabilities[i] = exps / np.sum(exps)
    return probabilities


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num_samples = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / num_samples
    return loss


if __name__ == "__main__":
    x, y = get_dataset()
    print(y)

    y_true = to_categorical(y, num_classes=2)
    print(y_true)

    y_logits = np.array([[10.8, -3.3], [12.2, 21.8], [1.1, 4.9], [7.05, 3.95]])

    y_pred = softmax(y_logits)
    print(y_pred)

    loss = cross_entropy(y_true, y_pred)
    print(loss)
