import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class MNIST:
    def __init__(self):
        # Load the data set
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # Convert to float32
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # Reshape the x data to shape (num_examples, width, height, depth)
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        # Save important data attributes as variables
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.num_classes = 10
        # Reshape the y data to one hot encoding
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        # # MinMaxScaler [0, 255] => [0.0, 1.0]
        # max_value = np.max(self.x_train)
        # self.x_train = self.x_train / max_value
        # self.x_test = self.x_test / max_value
        # # StandardScaler [0, 255] => mean=0, std=1
        # mean = np.mean(self.x_train)
        # std = np.std(self.x_train)
        # self.x_train = (self.x_train - mean) / std
        # self.x_test = (self.x_test - mean) / std


    def get_train_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test

if __name__ == "__main__":
    mnist = MNIST()

    x_train, y_train = mnist.get_train_set()
    print(np.max(x_train))
    print(np.min(x_train))
    print(np.mean(x_train))
    print(np.std(x_train))
