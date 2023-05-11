import numpy as np
import tensorflow as tf
from keras.layers import SimpleRNN
from keras.models import Sequential


np.random.seed(0)
tf.random.set_seed(0)


def sigmoid_fn(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh_fn(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class SimpleRNNInference:
    def __init__(
        self,
        rnn_layer: tf.keras.layers.SimpleRNN,
        return_sequences: bool = False,
    ) -> None:
        self.return_sequences = return_sequences
        self.rnn_layer = rnn_layer
        self.W, self.U, self.b = self.rnn_layer.get_weights()
        self.num_units = self.b.shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # output shape (num_timesteps, num_units)
        if self.return_sequences:
            time_steps = x.shape[0]
            self.h = np.zeros((time_steps, self.num_units))
        # output shape (num_units)
        else:
            self.h = np.zeros((self.num_units))
        h_t = np.zeros((1, self.num_units))
        for t, x_t in enumerate(x):
            x_t = x_t.reshape(1, -1)  # (2) => (1, 2)
            h_t = self.forward_step(x_t, h_t)
            if self.return_sequences:
                self.h[t] = h_t
            else:
                self.h = h_t
        return self.h

    def forward_step(self, x_t: np.ndarray, h_t: np.ndarray) -> np.ndarray:
        h_t = np.matmul(h_t, self.U) + np.matmul(x_t, self.W) + self.b
        h_t = tanh_fn(h_t)  # (-1, 1)
        return h_t


if __name__ == "__main__":
    # data set shape = (num_samples, num_timesteps, num_features)
    # input shape = (num_timesteps, num_features)
    # iöf return_sequences is True:
    #   output shape = (num_timesteps, num_units)
    # else:
    #   output shape = (1, num_units)
    x = np.random.normal(size=(1, 3, 2))
    num_units = 4
    return_sequences = True

    # num_features = 2
    # num_units = 4
    # h_t shape = (4),        (num_units)
    # W shape   = (2, 4),     (num_features, num_units)
    # U shape   = (4, 4),     (num_units, num_units)
    # b shape   = (4),        (num_units)
    #
    # matmul(x, W)      (1, 2)*(2,4) => (4)
    # matmul(h, U)      (1, 4)*(4,4) => (4)
    # intern + b        (4)+(4)   => (4)
    model = Sequential()
    model.add(
        SimpleRNN(
            units=num_units,
            return_sequences=return_sequences,
            input_shape=x.shape[1:],
        )
    )
    model.compile(loss="mse", optimizer="Adam")
    # model.summary()

    rnn = SimpleRNNInference(
        rnn_layer=model.layers[0], return_sequences=return_sequences
    )
    output_rnn_own = rnn(x[0])  # 10.5
    print(output_rnn_own)
    print("\n\n")
    output_rnn_tf = model.predict(x[[0]])
    print(output_rnn_tf)  # 10.5
    assert np.all(np.isclose(output_rnn_own - output_rnn_tf, 0.0, atol=1e-06))
