from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def f(
    x0: float | np.ndarray,
    x1: float | np.ndarray,
) -> float | np.ndarray:
    """Rosenbrock Function."""
    return 100.0 * (x0**2.0 - x1) ** 2.0 + (x0 - 1.0) ** 2.0


def f_prime_x0(
    x0: float | np.ndarray,
    x1: float | np.ndarray,
) -> float | np.ndarray:
    """Derivative of f w.r.t. x0."""
    return 2.0 * (200.0 * x0 * (x0**2.0 - x1) + x0 - 1.0)


def f_prime_x1(
    x0: float | np.ndarray,
    x1: float | np.ndarray,
) -> float | np.ndarray:
    """Derivative of f w.r.t. x0."""
    return -200.0 * (x0**2.0 - x1)


def plot_rosenbrock(
    x_start: np.ndarray,
    gradient_steps: list[Any] | None = None,
) -> None:
    """Plot the gradient steps."""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")

    s = 0.3
    X = np.arange(-2, 2.0 + s, s)  # noqa: N806
    Y = np.arange(-2, 3.0 + s, s)  # noqa: N806

    # Create the mesh grid(s) for all X/Y combos.
    X, Y = np.meshgrid(X, Y)  # noqa: N806
    # Rosenbrock function w/ two parameters using numpy Arrays
    Z = f(X, Y)  # noqa: N806

    surf = ax.plot_surface(  # type: ignore
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        linewidth=0,
        alpha=0.8,
        cmap=cm.coolwarm,  # type: ignore
    )
    # Global minimum
    ax.scatter(1, 1, f(1.0, 1.0), color="red", marker="*", s=200)  # type: ignore
    # Starting point
    x0, x1 = x_start
    ax.scatter(x0, x1, f(x0, x1), color="green", marker="o", s=200)  # type: ignore

    # Eps off set of the z axis, to plot the points above the surface for vis
    eps = 50
    if gradient_steps:
        for x0, x1 in gradient_steps:
            ax.scatter(  # type: ignore
                x0,
                x1,
                f(x0, x1) + eps,
                color="green",
                marker="o",
                s=25,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def main() -> None:
    x0 = np.random.uniform(-2.0, 2.0)  # noqa: NPY002
    x1 = np.random.uniform(-2.0, 2.0)  # noqa: NPY002
    x_start = np.array([x0, x1])
    plot_rosenbrock(x_start)


if __name__ == "__main__":
    main()
