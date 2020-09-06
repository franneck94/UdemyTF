import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[0]
image = image.reshape((28, 28))

kernel = np.random.uniform(low=0.0, high=1.0, size=(2, 2))


def conv2D(image, kernel):
    '''Conv Funktion definieren und anschließend plotten.'''
    rows, cols = image.shape  # 28x28
    k_size, _ = kernel.shape  # 2x2
    conv_image = np.zeros(shape=(rows, cols), dtype=np.float32)  # 28x28

    for i in range(rows - k_size):
        for j in range(cols - k_size):
            conv_image[i][j] = np.sum(kernel * image[i : i + k_size, j : j + k_size])

    return conv_image


conv_image = conv2D(image, kernel)

# Input und Outputbild des Pooling Layers mit imshow() ausgeben
plt.imshow(image, cmap="gray")
plt.show()

plt.imshow(conv_image, cmap="gray")
plt.show()
