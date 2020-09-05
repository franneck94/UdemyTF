import numpy as np


def get_dataset():
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return x, y


class Perceptron():
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.w = []
        self.lr = lr

    def train(self, x, y):
        N, dim = x.shape # 4x2
        # Init model
        self.w = np.random.uniform(-1, 1, (dim, 1)) # Gleichverteilung [-1, 1]: 2 Weights
        print(self.w)
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
        print("Train Error: ", error)
        print(self.w)

    def test(self, x, y):
        y_pred = np.array([self.predict(x_i) for x_i in x])
        acc = sum(1 for y_p, y_i in zip(y_pred, y) if y_p == y_i) / y.shape[0]
        print("Test Acc: ", acc)

    def update_weights(self, x, y, y_hat):
        for i in range(self.w.shape[0]):
            delta_w_i = self.lr * (y - y_hat) * x[i]
            self.w[i] = self.w[i] + delta_w_i

    # Hier soll der Treshold verändert werden.
    def activation(self, signal):
        if signal > 1: # Das ist der Threshold
            return 1
        else:
            return 0

    def predict(self, x):
        input_signal = np.dot(self.w.T, x)
        output_signal = self.activation(input_signal)
        return output_signal


x, y = get_dataset()

lr = 0.1
epochs = 100

p = Perceptron(epochs=epochs, lr=lr)
p.train(x, y)
p.test(x, y)
