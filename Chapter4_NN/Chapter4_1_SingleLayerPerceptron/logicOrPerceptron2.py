import numpy as np
import matplotlib.pyplot as plt

def get_dataset():
    x = np.array([[0,0], [1,0], [0,1], [1,1]])
    y = np.array([0, 1, 1, 1])
    return x, y

class Perceptron():
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.w = []
        self.lr = lr

    def train(self, x, y):
        N, dim = x.shape # 4x2
        # Init model
        self.w = np.random.uniform(-1, 1, (dim,1)) # Gleichverteilung [-1, 1]: 2 Weights
        # Training
        error = 0.0
        for epoch in range(self.epochs):
            choice = np.random.choice(N) # Pick random sample from dataset
            x_i = x[choice]
            y_i = y[choice]
            y_hat = self.predict(x_i)
            # Check if we made a false classification
            if y_hat != y_i:
                error += 1
                self.update_weights(x_i, y_i, y_hat)

    def update_weights(self, x, y, y_hat):
        for i in range(self.w.shape[0]):
            delta_w_i = self.lr * (y - y_hat) * x[i]
            self.w[i] = self.w[i] + delta_w_i

    def activation(self, signal):
        if signal > 0:
            return 1
        else:
            return 0

    def predict(self, x):
        input_signal = np.dot(self.w.T, x)
        output_signal = self.activation(input_signal)
        return output_signal