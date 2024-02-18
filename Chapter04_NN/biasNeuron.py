import matplotlib.pyplot as plt
import numpy as np


def plot_function(
    x: np.ndarray,
    y: np.ndarray,
    b: float = 0.0,
    name: str = "",
) -> None:
    plt.step(x + b, y, color="blue")
    plt.step(x, y, color="black")
    plt.step(x - b, y, color="red")
    plt.xlabel("a")
    plt.ylabel("f(a)")
    plt.legend([f"Bias: -{b}", "Bias: 0", f"Bias: {b}"])
    plt.title(f"Activation function: {name}")
    plt.show()


def main() -> None:
    x = np.linspace(
        start=-10,
        stop=10,
        num=1000,
    )
    b = 2

    # Tanh
    # f(a) = tanh(a) = (2 / (1+e^(-2a)) )- 1
    y = np.array(
        [(2 / (1 + np.exp(-2 * a))) - 1 for a in x],
    )
    plot_function(x, y, b, name="tanh")

    # SIGMOID
    # sigmoid(a) = 1 / (1 + e^(-a))
    y = np.array(
        [1 / (1 + np.exp(-a)) for a in x],
    )
    plot_function(x, y, b, name="sigmoid")

    # RELU = Rectified Linear Unit
    # f(a) = max (0, a)
    y = np.array(
        [max(0, a) for a in x],
    )
    plot_function(x, y, b, name="relu")


if __name__ == "__main__":
    main()
