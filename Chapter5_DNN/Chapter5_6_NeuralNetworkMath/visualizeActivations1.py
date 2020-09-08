import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# y = ReLU(wx + b)
# y = sigmoid(wx + b)
# shift = -b/w
w = 1
b = -4
# shift = 2
act = sigmoid

x = np.linspace(start=-10, stop=10, num=5000)
y_act = np.array([act(xi * w + b) for xi in x])
y = np.array([act(xi * 1 + 0) for xi in x])

plt.figure(figsize=(8, 5))
plt.grid(True)
plt.plot(x, y, color="blue")
plt.plot(x, y_act, color="red")
plt.show()
