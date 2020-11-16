import os

import matplotlib.pyplot as plt
import numpy as np


IMG_PATH = os.path.dirname(os.path.abspath(__file__))


def plot_function(
    input_signal: np.ndarray,
    output_signal_neg: np.ndarray,
    output_signal: np.ndarray,
    output_signal_pos: np.ndarray,
    name: str = None
) -> None:
    plt.step(input_signal, output_signal_neg, color="blue")
    plt.step(input_signal, output_signal, color="black")
    plt.step(input_signal, output_signal_pos, color="red")
    plt.xlabel("a")
    plt.ylabel("f(a)")
    plt.xlim(np.min(input_signal) - 0.2, np.max(input_signal) + 0.2)
    plt.ylim(np.min(output_signal) - 0.2, np.max(output_signal) + 0.2)
    plt.legend(["Bias: -2", "Bias: 0", "Bias: 2"])
    if name:
        plt.title(f"Activation function: {name}")
    plt.show()


if __name__ == "__main__":
    input_signal = np.linspace(start=-10, stop=10, num=1000)
    bias = 2

    # Step function
    # f(a) = 0, if a <= 0 else 1
    output_signal_neg = [0 if a - bias <= 0 else 1 for a in input_signal]
    output_signal = [0 if a <= 0 else 1 for a in input_signal]
    output_signal_pos = [0 if a + bias <= 0 else 1 for a in input_signal]
    plot_function(input_signal, output_signal_neg, output_signal, output_signal_pos, name="step")

    # Tanh
    # f(a) = tanh(a) = 2 / (1+e^(-2a)) - 1
    output_signal_neg = [2 / (1 + np.exp(-2 * a - bias)) - 1 for a in input_signal]
    output_signal = [2 / (1 + np.exp(-2 * a)) - 1 for a in input_signal]
    output_signal_pos = [2 / (1 + np.exp(-2 * a + bias)) - 1 for a in input_signal]
    plot_function(input_signal, output_signal_neg, output_signal, output_signal_pos, name="tanh")

    # SIGMOID
    # sigmoid(a) = 1 / (1 + e^-a)
    output_signal_neg = [1 / (1 + np.exp(-(a - bias))) for a in input_signal]
    output_signal = [1 / (1 + np.exp(-a)) for a in input_signal]
    output_signal_pos = [1 / (1 + np.exp(-(a + bias))) for a in input_signal]
    plot_function(input_signal, output_signal_neg, output_signal, output_signal_pos, name="sigmoid")

    # RELU = Rectified Linear Unit
    # f(a) = max (0, a)
    output_signal_neg = [max(0, a - bias) for a in input_signal]
    output_signal = [max(0, a) for a in input_signal]
    output_signal_pos = [max(0, a + bias) for a in input_signal]
    plot_function(input_signal, output_signal_neg, output_signal, output_signal_pos, name="relu")
