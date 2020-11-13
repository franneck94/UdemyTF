from typing import Tuple

import numpy as np


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """OR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    return x, y


def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y_one_hot = np.zeros(shape=(len(y), num_classes))  # 4x2
    for i, y_i in enumerate(y):
        y_oh = np.zeros(shape=num_classes)
        y_oh[y_i] = 1
        y_one_hot[i] = y_oh
    return y_one_hot


def softmax(y_pred: np.ndarray) -> np.ndarray:
    y_softmax = np.zeros(shape=y_pred.shape)
    for i in range(len(y_pred)):
        exps = np.exp(y_pred[i])
        y_softmax[i] = exps / np.sum(exps)
    return y_softmax


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num_samples = y_pred.shape[0]
    num_classes = y_pred.shape[1]
    loss = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        for c in range(num_classes):
            loss -= y_t[c] * np.log(y_p[c])
    return loss / num_samples


if __name__ == "__main__":
    x, y = get_dataset()
    y = to_one_hot(y, num_classes=2)
    print(y)

    p1 = np.array([0.223, 0.613])
    p2 = np.array([-0.750, 0.500])
    p3 = np.array([0.010, 0.200])
    p4 = np.array([0.564, 0.234])
    y_pred = np.array([p1, p2, p3, p4])

    print(y_pred)
    y_pred = softmax(y_pred)
    print(y_pred)

    loss = cross_entropy(y, y_pred)
    print(loss)
