import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def f(x):
    return x**4 + -5*x**3 + 14*x**2 + x + 10

x = np.linspace(start=-10.0, stop=10.0, num=2000).reshape(-1, 1)
y = f(x)

def relu(x):
    if x > 0: return x
    else: return 0

model = Sequential()
model.add(Dense(200)) # Input zu Hidden
model.add(Dense(200)) # Input zu Hidden
model.add(Dense(1)) # Vom Hidden zum Output
model.compile(optimizer=Adam(lr=1e-2), loss="mse")
model.fit(x, y, epochs=30)
y_pred_linear = model.predict(x)

model = Sequential()
model.add(Dense(500)) # Input zu Hidden
model.add(Activation("relu")) # ReLU vom Hidden
model.add(Dense(500)) # Input zu Hidden
model.add(Activation("relu")) # ReLU vom Hidden
model.add(Dense(1)) # Vom Hidden zum Output
model.compile(optimizer=Adam(lr=1e-2), loss="mse")
model.fit(x, y, epochs=30)
y_pred_relu = model.predict(x)

model = Sequential()
model.add(Dense(500)) # Input zu Hidden
model.add(Activation("sigmoid")) # ReLU vom Hidden
model.add(Dense(500)) # Input zu Hidden
model.add(Activation("sigmoid")) # ReLU vom Hidden
model.add(Dense(1)) # Vom Hidden zum Output
model.compile(optimizer=Adam(lr=1e-2), loss="mse")
model.fit(x, y, epochs=30)
y_pred_sigmoid = model.predict(x)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(24,12))
plt.grid(True)
ax1.plot(x, y, color="blue")
ax1.plot(x.flatten(), y_pred_linear.flatten(), color="red")
ax2.plot(x, y, color="blue")
ax2.plot(x.flatten(), y_pred_sigmoid.flatten(), color="red")
ax3.plot(x, y, color="blue")
ax3.plot(x.flatten(), y_pred_relu.flatten(), color="red")
plt.show()