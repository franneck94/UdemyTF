import numpy as np
import tensorflow as tf


def get_dataset():
    """XOR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(np.float32)
    y = np.array([0, 1, 1, 0]).astype(np.float32)
    return x, y


x, y = get_dataset()
x_train, y_train = x, y
x_test, y_test = x, y

# Dataset variables
features = 2
classes = 2
target = 1
# Model variables
hidden_layer_size = 2
nodes = [features, hidden_layer_size, target]  # input, hidden, output
train_size = x_train.shape[0]
test_size = x_test.shape[0]
epochs = 10


class Model:
    def __init__(self):
        # Create weights and biases
        # Weights (Matrices)
        # Input to Hidden Layer
        self.W1 = tf.Variable(tf.random.uniform(shape=[nodes[0], nodes[1]], minval=-1.0, maxval=1.0))
        # Hidden to Output Layer
        self.W2 = tf.Variable(tf.random.uniform(shape=[nodes[1], nodes[2]], minval=-2.0, maxval=2.0))
        # Biases (Vectors)
        # Bias vector from the input layer to the hidden layer
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]))
        # Bias vector from the hidden layer to the output layer
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]))
        self.variables = [self.W1, self.W2, self.b1, self.b2]

    def predict(self, x):
        # Input layer
        input_layer = x
        # From the input to the hidden layer
        hidden_layer = tf.math.add(tf.linalg.matmul(input_layer, self.W1), self.b1)
        # ReLU = max(val, 0.0)
        hidden_layer_act = tf.nn.relu(hidden_layer)
        # From the hidden to the output layer
        output_layer = tf.math.add(tf.linalg.matmul(hidden_layer_act, self.W2), self.b2)
        # sigmoid = 1 / (1 + exp(-x))
        output_layer_act = tf.nn.sigmoid(output_layer)
        return output_layer_act


model = Model()
y_pred = model.predict(x_train)
print(y_train, y_pred.numpy())
