import numpy as np
from tensorflow.keras.datasets import mnist

from plotting import display_digit


# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_size = x_train.shape[0]
test_size = x_test.shape[0]

print(train_size)
print(test_size)

print(x_train.shape)
print(y_train.shape)

for i in np.random.randint(train_size, size=10):
    display_digit(x_train[i], label=y_train[i])
