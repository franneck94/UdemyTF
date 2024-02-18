import matplotlib.pyplot as plt
import numpy as np


def plot_function(
    x: np.ndarray,
    y: np.ndarray,
    name: str = "",
) -> None:
    plt.step(x, y)
    plt.xlabel("a")
    plt.ylabel("f(a)")
    plt.title(f"Activation function: {name}")
    plt.show()


def main() -> None:
    x = np.linspace(
        start=-10,
        stop=10,
        num=1000,
    )

    # Step function
    # f(a) = 0, if a <= 0 else 1
    # {0, 1}
    y = np.array(
        [0 if a <= 0 else 1 for a in x],
    )
    plot_function(x, y, name="step")

    # Tanh
    # f(a) = tanh(a) = (2 / (1+e^(-2a))) - 1
    # [-1, 1]
    y = np.array(
        [(2 / (1 + np.exp(-2 * a))) - 1 for a in x],
    )
    plot_function(x, y, name="tanh")

    # SIGMOID
    # sigmoid(a) = 1 / (1 + e^-a)
    # [0, 1]
    y = np.array(
        [1 / (1 + np.exp(-a)) for a in x],
    )
    plot_function(x, y, name="sigmoid")

    # RELU = Rectified Linear Unit
    # f(a) = max (0, a)
    # [0, inf]
    y = np.array(
        [max(0, a) for a in x],
    )
    plot_function(x, y, name="relu")


if __name__ == "__main__":
    main()
