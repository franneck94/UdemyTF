from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """XOR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return x, y


def dense(W: tf.Variable, b: tf.Variable, x: tf.Tensor) -> tf.Tensor:
    """output = W*x + b"""
    return tf.math.add(tf.linalg.matmul(x, W), b)


class Model:
    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
        metric: tf.keras.metrics.Metric,
        units_list: List[int]
    ) -> None:
        # Model parameters
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.units_list = units_list
        # Weights
        W1_shape = [self.units_list[0], self.units_list[1]]
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=W1_shape, stddev=0.1), name="W1")
        W2_shape = [self.units_list[1], self.units_list[2]]
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=W2_shape, stddev=0.1), name="W2")
        # Biases
        b1_shape = [self.units_list[1]]
        self.b1 = tf.Variable(tf.constant(0.0, shape=b1_shape), name="b1")
        b2_shape = [self.units_list[2]]
        self.b2 = tf.Variable(tf.constant(0.0, shape=b2_shape), name="b2")
        # Trainable variables
        self.variables = [self.W1, self.W2, self.b1, self.b2]

    def _update_weights(self, x: np.ndarray, y: np.ndarray) -> tf.Tensor:
        with tf.GradientTape() as tape:
            y_pred = self.predict(x)
            loss_value = self.loss(y, y_pred)
        gradients = tape.gradient(loss_value, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss_value

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            # Loss value
            loss_value = self._update_weights(x, y).numpy()
            # Metric value
            y_pred = self.predict(x)
            self.metric.reset_states()
            self.metric.update_state(y, y_pred)
            metric_value = self.metric.result().numpy()
            print(f"Epoch: {epoch} - Loss: {loss_value} - Metric: {metric_value}")

    def predict(self, x: np.ndarray) -> tf.Tensor:
        input_layer = x
        hidden_layer = dense(self.W1, self.b1, input_layer)
        hidden_layer_activation = tf.nn.tanh(hidden_layer)
        output_layer = dense(self.W2, self.b2, hidden_layer_activation)
        output_layer_activation = tf.nn.sigmoid(output_layer)
        return output_layer_activation

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        y_pred = self.predict(x)
        # Loss value
        loss_value = self.loss(y, y_pred).numpy()
        # Metric value
        self.metric.reset_states()
        self.metric.update_state(y, y_pred)
        metric_value = self.metric.result().numpy()
        return [loss_value, metric_value]


if __name__ == "__main__":
    x, y = get_dataset()

    num_features = 2
    num_targets = 1
    units_list = [num_features, 6, num_targets]

    learning_rate = 0.5
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsoluteError()
    metric = tf.keras.metrics.BinaryAccuracy()

    model = Model(optimizer, loss, metric, units_list)
    model.fit(x, y, epochs=100)
