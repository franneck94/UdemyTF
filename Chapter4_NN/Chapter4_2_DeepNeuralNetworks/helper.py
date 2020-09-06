import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D


def f(x0, x1):
    '''Rosenbrock Funktion.'''
    return 100 * (x0 ** 2 - x1) ** 2 + (x0 - 1) ** 2


def f_prime_x0(x0, x1):
    '''Derivative of x0.'''
    return 2 * (200 * x0 * (x0 ** 2 - x1) + x0 - 1)


def f_prime_x1(x0, x1):
    '''Derivative of x0.'''
    return -200 * (x0 ** 2 - x1)


def plot_rosenbrock(x_start, gradient_steps=None):
    '''Plot the gradient steps.'''
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig)

    s = 0.3
    X = np.arange(-2, 2.0 + s, s)
    Y = np.arange(-2, 3.0 + s, s)

    # Create the mesh grid(s) for all X/Y combos.
    X, Y = np.meshgrid(X, Y)
    # Rosenbrock function w/ two parameters using numpy Arrays
    Z = f(X, Y)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=0.8, cmap=cm.coolwarm)
    # Global minimum
    ax.scatter(1, 1, f(1, 1), color="red", marker="*", s=200)
    # Starting point
    x0, x1 = x_start
    ax.scatter(x0, x1, f(x0, x1), color="green", marker="o", s=200)

    # Eps off set of the z axis, to plot the points above the surface for better visualization
    eps = 50
    if gradient_steps:
        for (x0, x1) in gradient_steps:
            ax.scatter(x0, x1, f(x0, x1) + eps, color="green", marker="o", s=50)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
