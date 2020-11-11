import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


def conv2D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return image  # TODO


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image = x_train[0]
    image = image.reshape((28, 28))
    kernel = np.random.uniform(low=0.0, high=1.0, size=(2, 2))

    conv_image = conv2D(image, kernel)

    plt.imshow(image, cmap="gray")
    plt.show()

    plt.imshow(conv_image, cmap="gray")
    plt.show()
