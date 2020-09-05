import os

import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = os.path.dirname(os.path.abspath(__file__))

# Step function
# f(a) = 0, if a <= 0 else 1
data = [0 if a <= 0 else 1 for a in np.linspace(start=-10, stop=10, num=1000)]

plt.step(np.linspace(start=-10, stop=10, num=1000), data)
plt.xlabel('a')
plt.ylabel('step(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 1.1)
#plt.savefig(os.path.join(IMG_PATH, "step.png"))
plt.show()

# Tanh
# f(a) = tanh(a) = 2 / (1+e^(-2a)) - 1
data = [2 / (1 + np.exp(-2 * a)) - 1 for a in np.linspace(start=-10, stop=10, num=1000)]

plt.plot(np.linspace(start=-10, stop=10, num=1000), data)
plt.xlabel('a')
plt.ylabel('tanh(a)')
plt.xlim(-12, 12)
plt.ylim(-1.1, 1.1)
#plt.savefig(os.path.join(IMG_PATH, "tanh.png"))
plt.show()

# SIGMOID
# sigmoid(a) = 1 / (1 + e^-a)
data = [1 / (1 + np.exp(-a)) for a in np.linspace(start=-10, stop=10, num=1000)]

plt.plot(np.linspace(start=-10, stop=10, num=1000), data)
plt.xlabel('a')
plt.ylabel('sigmoid(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 1.1)
#plt.savefig(os.path.join(IMG_PATH, "sigmoid.png"))
plt.show()

# RELU = Rectified Linear Unit
# f(a) = max (0, a)
data = [max(0, a) for a in np.linspace(start=-10, stop=10, num=1000)]

plt.plot(np.linspace(start=-10, stop=10, num=1000), data)
plt.xlabel('a')
plt.ylabel('relu(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 10)
#plt.savefig(os.path.join(IMG_PATH, "relu.png"))
plt.show()
