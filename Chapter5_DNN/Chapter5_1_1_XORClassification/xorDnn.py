import numpy as np
import tensorflow as tf


def get_dataset():
    """XOR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return x, y


def accuracy_score(y_true, y_pred):
    y_pred_class = tf.reshape(tf.cast(tf.math.greater(y_pred, 0.5), tf.float32), y_true.shape)
    correct_result = tf.math.equal(y_pred_class, y_true)
    accuracy = tf.math.reduce_mean(tf.cast(correct_result, tf.float32))
    return accuracy


def mse_loss(y_true, y_pred):
    """MSE = 1/N * Sum( (y_true - y_pred)^2)"""
    loss_fn = tf.math.reduce_mean(tf.math.square(y_pred - y_true))
    return loss_fn


def dense(W, b, x):
    return tf.math.add(tf.linalg.matmul(x, W), b)


class Model:
    def __init__(self, optimizer, loss, metric):
        # Weights (Matrices)
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=[nodes[0], nodes[1]], stddev=0.1), name="W1",)
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=[nodes[1], nodes[2]], stddev=0.1), name="W2",)
        # Biases (Vectors)
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]), name="b1")
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]), name="b2")
        self.variables = [self.W1, self.W2, self.b1, self.b2]
        # Model variables
        self.optimizer = optimizer
        self.loss: tf.keras.losses.Loss = loss
        self.metric: tf.keras.metrics.Metric = metric

    def _get_variables(self):
        return {var.name: var.numpy() for var in self.variables}

    def _update_variables(self, x_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_train)
            loss_val = self.loss(y_train, y_pred)
        gradients = tape.gradient(loss_val, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss_val

    def fit(self, x_train, y_train, epochs=10):
        for epoch in range(1, epochs + 1):
            # Loss
            train_loss = self._update_variables(x_train, y_train).numpy()
            # Metrics
            y_pred = self.predict(x_train)
            self.metric.reset_states()
            self.metric.update_state(y_train, y_pred)
            train_accuracy = self.metric.result().numpy()
            print(f"Epoch: {epoch} - Train Loss: {train_loss} - Train Acc: {train_accuracy}")

    def predict(self, x):
        input_layer = x
        hidden_layer = dense(self.W1, self.b1, input_layer)
        hidden_layer_act = tf.nn.tanh(hidden_layer)
        output_layer = dense(self.W2, self.b2, hidden_layer_act)
        output_layer_act = tf.nn.sigmoid(output_layer)
        return output_layer_act

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        loss_val = self.loss(y, y_pred).numpy()
        accuracy_val = self.metric(y, y_pred).numpy()
        return [loss_val, accuracy_val]


if __name__ == "__main__":
    x, y = get_dataset()

    # Dataset variables
    features = 2
    classes = 2
    target = 1
    # Model variables
    hidden_layer_size = 6
    nodes = [features, hidden_layer_size, target]  # input, hidden, output
    epochs = 100

    learning_rate = 0.1
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    metric = tf.keras.metrics.BinaryAccuracy()

    model = Model(optimizer, loss, metric)
    model.fit(x, y, epochs=epochs)
    model.evaluate(x, y)
