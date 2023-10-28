import matplotlib.pyplot as plt
import numpy as np


def relu(x: float) -> float:
    if x > 0:
        return x
    return 0


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def main() -> None:
    # y = ReLU(wx + b)
    # y = sigmoid(wx + b)
    # shift = -b/w
    w = 2
    b = -4
    # shift = 2
    act_fn = relu

    x = np.linspace(start=-10, stop=10, num=500)
    y_act = np.array([act_fn(xi * w + b) for xi in x])
    y = np.array([act_fn(xi * 1 + 0) for xi in x])

    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.plot(x, y, color="blue")
    plt.plot(x, y_act, color="red")
    plt.ylim(0.0, 50.0)
    plt.show()


if __name__ == "__main__":
    main()
