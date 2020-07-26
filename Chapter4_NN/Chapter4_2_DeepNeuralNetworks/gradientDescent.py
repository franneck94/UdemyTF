import numpy as np

from helper import f, f_prime_x0, f_prime_x1, plot_rosenbrock

x0 = np.random.uniform(-2, 2)
x1 = np.random.uniform(-2, 2)
x_start = (x0, x1)
y = f(x0, x1)

print("\n\nGlobale Minimum bei: ", 1, 1)
print("Starte bei x = ", x_start)
print("Mit f(x) = ", y)
plot_rosenbrock(x_start)

eta = 0.005
it = 0
stop_iter = 1000

gradient_steps = []

while it < stop_iter:
    x0 = x0 - eta * f_prime_x0(x0, x1)
    x1 = x1 - eta * f_prime_x1(x0, x1)
    it += 1
    fx = f(x0, x1)
    if it % 100 == 0:
        gradient_steps.append((x0, x1))

print("Solution: ", fx)
print("x0 = ", x0)
print("x1 = ", x1)
plot_rosenbrock(x_start=x_start, gradient_steps=gradient_steps)
