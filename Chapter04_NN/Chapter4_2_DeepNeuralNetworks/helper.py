from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def f(
    x0: Union[float, np.ndarray], x1: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Rosenbrock Function."""
    result = 100.0 * (x0**2.0 - x1) ** 2.0 + (x0 - 1.0) ** 2.0
    return result


def f_prime_x0(
    x0: Union[float, np.ndarray], x1: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Derivative of f w.r.t. x0."""
    result = 2.0 * (200.0 * x0 * (x0**2.0 - x1) + x0 - 1.0)
    return result


def f_prime_x1(
    x0: Union[float, np.ndarray], x1: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Derivative of f w.r.t. x0."""
    result = -200.0 * (x0**2.0 - x1)
    return result


def plot_rosenbrock(x_start: np.ndarray, gradient_steps: list = None) -> None:
    """Plot the gradient steps."""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")

    s = 0.3
    X = np.arange(-2, 2.0 + s, s)
    Y = np.arange(-2, 3.0 + s, s)

    # Create the mesh grid(s) for all X/Y combos.
    X, Y = np.meshgrid(X, Y)
    # Rosenbrock function w/ two parameters using numpy Arrays
    Z = f(X, Y)  # type: ignore

    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=0.8, cmap=cm.coolwarm
    )
    # Global minimum
    ax.scatter(1, 1, f(1.0, 1.0), color="red", marker="*", s=200)
    # Starting point
    x0, x1 = x_start
    ax.scatter(x0, x1, f(x0, x1), color="green", marker="o", s=200)

    # Eps off set of the z axis, to plot the points above the surface for vis
    eps = 50
    if gradient_steps:
        for x0, x1 in gradient_steps:
            ax.scatter(x0, x1, f(x0, x1) + eps, color="green", marker="o", s=25)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def main() -> None:
    x0 = np.random.uniform(-2.0, 2.0)
    x1 = np.random.uniform(-2.0, 2.0)
    x_start = np.array([x0, x1])
    plot_rosenbrock(x_start)


if __name__ == "__main__":
    main()
