from typing import Tuple

import numpy as np


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """OR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    return x, y


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    N = y_true.shape[0]
    accuracy = np.sum(y_true == y_pred) / N
    return accuracy


def step_function(input_signal: np.ndarray) -> np.ndarray:
    output_signal = (input_signal > 0.0).astype(np.int_)
    return output_signal


class Perceptron:
    def __init__(self) -> None:
        self.learning_rate: float
        self.w: np.ndarray

    def compile(self, learning_rate: float, input_dim: int) -> None:
        self.learning_rate = learning_rate
        self.w = np.random.uniform(-1, 1, size=(input_dim, 1))

    def _update_weights(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        error = (y - y_pred)
        delta = error * x
        for delta_i in delta:
            self.w = self.w + self.learning_rate * delta_i.reshape(-1, 1)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            y_pred = self.predict(x)
            self._update_weights(x, y, y_pred)
            accuracy = accuracy_score(y, y_pred)
            print(f"Epoch: {epoch} Train Accuracy: {accuracy}")
            if accuracy == 1.0:
                break

    def predict(self, x: np.ndarray) -> np.ndarray:
        input_signal = np.dot(x, self.w)
        output_signal = step_function(input_signal)
        return output_signal

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)


if __name__ == "__main__":
    x, y = get_dataset()

    input_dim = x.shape[1]
    learning_rate = 0.5
    epochs = 10

    p = Perceptron()
    p.compile(learning_rate=learning_rate, input_dim=input_dim)
    p.train(x, y, epochs=10)
    score = p.evaluate(x, y)
    print(f"Final Accuracy: {score}")
