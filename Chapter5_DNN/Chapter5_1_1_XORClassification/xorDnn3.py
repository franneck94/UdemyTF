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
        # Weights (Matrices)
        self.W1 = tf.Variable(tf.random.uniform(shape=[nodes[0], nodes[1]], minval=-2.0, maxval=2.0))
        self.W2 = tf.Variable(tf.random.uniform(shape=[nodes[1], nodes[2]], minval=-2.0, maxval=2.0))
        # Biases (Vectors)
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]))
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]))
        self.variables = [self.W1, self.W2, self.b1, self.b2]
        # Model variables
        self.learning_rate = 0.001
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        self.current_loss_val = None

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

    def loss(self, y_true, y_pred):
        loss_fn = tf.math.reduce_mean(tf.math.square(y_pred - y_true))  # 1/N * Sum( (y_true - y_pred)^2)
        self.current_loss_val = loss_fn.numpy()
        return loss_fn

    def fit(self, x_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_train)
            loss = self.loss(y_train, y_pred)
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        y_pred_class = tf.reshape(tf.cast(tf.math.greater(y_pred, 0.5), tf.float32), y_train.shape)
        correct_result = tf.math.equal(y_pred_class, y_train)
        accuracy = tf.math.reduce_mean(tf.cast(correct_result, tf.float32))


model = Model()
model.fit(x_train, y_train)
