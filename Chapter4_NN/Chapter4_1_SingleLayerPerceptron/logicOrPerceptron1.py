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