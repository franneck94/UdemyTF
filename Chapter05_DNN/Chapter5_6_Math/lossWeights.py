import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error


def f(x: np.ndarray) -> np.ndarray:
    return x**2 + x + 10


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

    model.compile(optimizer=RMSprop(learning_rate=1e-2), loss="mse")

    model.fit(x, y, epochs=20)

    y_pred = model.predict(x).flatten()
    W, b = model.layers[0].get_weights()  # noqa: N806
    print(f"Weights: {W[0][0]}")

    w = np.linspace(start=-5, stop=5, num=200)
    losses = []
    for wi in w:
        W, b = model.layers[0].get_weights()  # noqa: N806
        W[0][0] = wi
        model.layers[0].set_weights((W, b))
        new_pred = model.predict(x).flatten()
        loss = mean_squared_error(y, new_pred)
        losses.append(loss)

    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax1.grid(True)
    ax2.grid(True)
    ax1.plot(x, y, color="blue")
    ax1.plot(x, y_pred, color="red")
    ax2.plot(w, losses, color="orange")
    plt.show()


if __name__ == "__main__":
    main()
