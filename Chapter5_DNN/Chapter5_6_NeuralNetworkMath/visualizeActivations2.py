from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def f(x: float) -> float:
    return x ** 2 + x + 10


def relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(start=-10.0, stop=10.0, num=1000).reshape(-1, 1)
    y = f(x)
    return x, y


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(12))  # Input zu Hidden
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(1))  # Vom Hidden zum Output
    return model


if __name__ == "__main__":
    x, y = get_dataset()

    model = build_model()

    model.compile(
        optimizer=Adam(learning_rate=5e-2),
        loss="mse"
    )
    model.fit(x, y, epochs=20)

    W, b = model.layers[0].get_weights()
    W2, b2 = model.layers[2].get_weights()
    W = W.flatten()
    W2 = W2.flatten()
    b = b.flatten()
    b2 = b2.flatten()
    print(W.shape, b.shape)
    print(W2.shape, b2.shape)

    # [1, 2, ...., 12]
    for i in range(1, len(W) + 1):
        y_hidden = np.array([W[:i] * xi + b[:i] for xi in x])
        y_relu = np.array([[relu(yhi) for yhi in yh] for yh in y_hidden])
        y_output = np.array([np.dot(W2[:i], yri) + b2 for yri in y_relu])

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        plt.title("Num weights: " + str(i))
        plt.grid(True)
        ax1.plot(x, y, color="blue")
        ax1.plot(x.flatten(), y_output.flatten(), color="red")
        ax2.plot(x, y_relu.T[-1])
        plt.show()
        plt.close()
