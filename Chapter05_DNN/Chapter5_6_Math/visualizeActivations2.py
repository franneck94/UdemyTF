import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def f(x: np.ndarray) -> np.ndarray:
    return x**2 + x + 10


def relu(x: np.ndarray) -> np.ndarray:
    if x > 0:
        return x
    return np.zeros_like(x)


def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(start=-10.0, stop=10.0, num=1000).reshape(-1, 1)
    y = f(x)
    return x, y


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(units=12))  # Input zu Hidden
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(units=1))  # Vom Hidden zum Output
    return model


def main() -> None:
    x, y = get_dataset()

    model = build_model()

    model.compile(optimizer=Adam(learning_rate=5e-2), loss="mse")
    model.fit(x, y, epochs=20)

    W, b = model.layers[0].get_weights()  # noqa: N806
    W2, b2 = model.layers[2].get_weights()  # noqa: N806
    W = W.flatten()  # noqa: N806
    W2 = W2.flatten()  # noqa: N806
    b = b.flatten()
    b2 = b2.flatten()
    print(W.shape, b.shape)
    print(W2.shape, b2.shape)

    # [1, 2, ...., 12]
    for i in range(1, len(W) + 1):
        y_hidden = np.array([W[:i] * xi + b[:i] for xi in x])
        y_relu = np.array([[relu(yhi) for yhi in yh] for yh in y_hidden])
        y_output = np.array([np.dot(W2[:i], yri) + b2 for yri in y_relu])

        _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        plt.title("Num weights: " + str(i))
        plt.grid(True)
        ax1.plot(x, y, color="blue")
        ax1.plot(x.flatten(), y_output.flatten(), color="red")
        ax2.plot(x, y_relu.T[-1])
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
