import os

import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = os.path.dirname(os.path.abspath(__file__))

# Bias neuron
b_0 = 1
w_0 = 2.0
THRESHOLD = b_0 * w_0

# plot function
# f(a) = 0, if x < 0 else 1
data_neg2 = [0 if (a - THRESHOLD) <= 0 else 1 for a in np.linspace(start=-10, stop=10, num=1000)]
data = [0 if a <= 0 else 1 for a in np.linspace(start=-10, stop=10, num=1000)]
data_pos2 = [0 if (a + THRESHOLD) <= 0 else 1 for a in np.linspace(start=-10, stop=10, num=1000)]

plt.step(np.linspace(start=-10, stop=10, num=1000), data_neg2, color='blue')
plt.step(np.linspace(start=-10, stop=10, num=1000), data, color='black')
plt.step(np.linspace(start=-10, stop=10, num=1000), data_pos2, color='red')
plt.xlabel('a')
plt.ylabel('step(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 1.1)
plt.legend(['Verschoben -2', 'Normal', 'Verschoben +2'])
#plt.savefig(os.path.join(IMG_PATH, "step2.png"))
plt.show()

# Tanh
# f(a) = tanh(a) = 2 / (1+e^(-2x)) - 1
data_neg = [2 / (1 + np.exp(-2 * (a - THRESHOLD) )) - 1 for a in np.linspace(start=-10, stop=10, num=1000)]
data = [2 / (1 + np.exp(-2 * a )) - 1 for a in np.linspace(start=-10, stop=10, num=1000)]
data_pos = [2 / (1 + np.exp(-2 * (a + THRESHOLD) )) - 1 for a in np.linspace(start=-10, stop=10, num=1000)]

plt.plot(np.linspace(start=-10, stop=10, num=1000), data_neg, color='blue')
plt.plot(np.linspace(start=-10, stop=10, num=1000), data, color='black')
plt.plot(np.linspace(start=-10, stop=10, num=1000), data_pos, color='red')
plt.xlabel('a')
plt.ylabel('tanh(a)')
plt.xlim(-12, 12)
plt.ylim(-1.1, 1.1)
plt.legend(['Verschoben -2', 'Normal', 'Verschoben +2'])
#plt.savefig(os.path.join(IMG_PATH, "tanh.png"))
plt.show()

# SIGMOID
# sigmoid(a) = 1 / (1 + e^-a)
data_neg = [1 / (1 + np.exp((-a - THRESHOLD))) for a in np.linspace(start=-10, stop=10, num=1000)]
data = [1 / (1 + np.exp(-a)) for a in np.linspace(start=-10, stop=10, num=1000)]
data_pos = [1 / (1 + np.exp((-a + THRESHOLD))) for a in np.linspace(start=-10, stop=10, num=1000)]

plt.plot(np.linspace(start=-10, stop=10, num=1000), data_neg, color='blue')
plt.plot(np.linspace(start=-10, stop=10, num=1000), data, color='black')
plt.plot(np.linspace(start=-10, stop=10, num=1000), data_pos, color='red')
plt.xlabel('a')
plt.ylabel('sigmoid(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 1.1)
plt.legend(['Verschoben -2', 'Normal', 'Verschoben +2'])
#plt.savefig(os.path.join(IMG_PATH, "sigmoid.png"))
plt.show()

# RELU = Rectified Linear Unit
# f(a) = max (0, a)

data_neg = [max(0, (a - THRESHOLD)) for a in np.linspace(start=-10, stop=10, num=1000)]
data = [max(0, a) for a in np.linspace(start=-10, stop=10, num=1000)]
data_pos = [max(0, (a + THRESHOLD)) for a in np.linspace(start=-10, stop=10, num=1000)]

plt.plot(np.linspace(start=-10, stop=10, num=1000), data_neg, color='blue')
plt.plot(np.linspace(start=-10, stop=10, num=1000), data, color='black')
plt.plot(np.linspace(start=-10, stop=10, num=1000), data_pos, color='red')
plt.xlabel('a')
plt.ylabel('relu(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 12.0)
plt.legend(['Verschoben -2', 'Normal', 'Verschoben +2'])
#plt.savefig(os.path.join(IMG_PATH, "relu2.png"))
plt.show()
