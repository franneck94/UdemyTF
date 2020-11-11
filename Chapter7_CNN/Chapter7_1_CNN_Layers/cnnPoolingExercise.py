import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


def max_pooling(image: np.ndarray) -> np.ndarray:
    return image


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image = x_train[0]
    image = image.reshape((28, 28))

    pooling_image = max_pooling(image)

    print(image.shape)
    print(pooling_image.shape)

    plt.imshow(image, cmap="gray")
    plt.show()

    plt.imshow(pooling_image, cmap="gray")
    plt.show()
