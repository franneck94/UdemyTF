import numpy as np

from helper import f, f_prime_x0, f_prime_x1, plot_rosenbrock


x0 = np.random.uniform(-2, 2)
x1 = np.random.uniform(-2, 2)
x_start = (x0, x1)
y_start = f(x0, x1)

print("Global minimum: ", 1, 1)
print("X_start = ", x_start)
print("Y_start = ", y_start)
plot_rosenbrock(x_start)

learning_rate = 0.005  # [0.001, 0.00001]
num_iterations = 1000

gradient_steps = []

for it in range(num_iterations):
    x0 = x0 - learning_rate * f_prime_x0(x0, x1)
    x1 = x1 - learning_rate * f_prime_x1(x0, x1)
    y = f(x0, x1)
    if it % 100 == 0:
        print("x0 = ", x0, " x1 = ", x1, " y = ", y)
        gradient_steps.append((x0, x1))

x_end = (x0, x1)
y_end = f(x0, x1)
print("X_end = ", x_end)
print("Y_end = ", y_end)
plot_rosenbrock(x_start=x_start, gradient_steps=gradient_steps)
