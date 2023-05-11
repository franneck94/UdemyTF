import numpy as np

from helper import f
from helper import f_prime_x0
from helper import f_prime_x1
from helper import plot_rosenbrock


np.random.seed(0)


def main() -> None:
    x0 = np.random.uniform(-2.0, 2.0)
    x1 = np.random.uniform(-2.0, 2.0)

    x_start = np.array([x0, x1])
    y_start = f(x0, x1)

    print(f"Global minimum: {(1, 1)}")
    print(f"X_start: {(x0, x1)}")
    print(f"Y_start: {y_start}")
    plot_rosenbrock(x_start)

    learning_rate = 0.005
    num_iterations = 1000

    gradient_steps = []

    for it in range(num_iterations):
        x0 = x0 - f_prime_x0(x0, x1) * learning_rate
        x1 = x1 - f_prime_x1(x0, x1) * learning_rate
        y = f(x0, x1)
        if it % 10 == 0:
            print(f"x0, x1 = {(x0, x1)}, y = {y}")
            gradient_steps.append((x0, x1))

    x_end = (x0, x1)
    y_end = f(x0, x1)
    print(f"x0 end, x1 end = {(x_end)}, y end = {y_end}")
    plot_rosenbrock(x_start, gradient_steps)


if __name__ == "__main__":
    main()
