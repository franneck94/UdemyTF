import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras.models import Sequential


np.random.seed(0)
tf.random.set_seed(0)


def sigmoid_fn(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh_fn(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class LSTMInference:
    def __init__(
        self,
        lstm_layer: tf.keras.layers.LSTM,
        return_sequences: bool = False,
    ) -> None:
        self.return_sequences = return_sequences
        self.lstm_layer = lstm_layer
        self.W, self.U, self.b = self.lstm_layer.get_weights()
        self.units = self.b.shape[0] // 4
        self.W_i = self.W[:, : self.units]
        self.W_f = self.W[:, self.units : self.units * 2]
        self.W_c = self.W[:, self.units * 2 : self.units * 3]
        self.W_o = self.W[:, self.units * 3 :]
        self.U_i = self.U[:, : self.units]
        self.U_f = self.U[:, self.units : self.units * 2]
        self.U_c = self.U[:, self.units * 2 : self.units * 3]
        self.U_o = self.U[:, self.units * 3 :]
        self.b_i = self.b[: self.units]
        self.b_f = self.b[self.units : self.units * 2]
        self.b_c = self.b[self.units * 2 : self.units * 3]
        self.b_o = self.b[self.units * 3 :]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # output shape (num_timesteps, units)
        if self.return_sequences:
            self.time_steps = x.shape[0]
            self.h = np.zeros((self.time_steps, self.units))
        # output shape (units)
        else:
            self.h = np.zeros(self.units)
        h_t = np.zeros((1, self.units))
        c_t = np.zeros((1, self.units))
        for t, x_t in enumerate(x):
            x_t = x_t.reshape(1, -1)  # (2) => (1, 2)
            c_t, h_t = self.forward_step(x_t, c_t, h_t)
            if self.return_sequences:
                self.h[t] = h_t
            else:
                self.h = h_t
        return self.h

    def forward_step(
        self,
        x_t: np.ndarray,
        c_t: np.ndarray,
        h_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        i_t = sigmoid_fn(
            np.matmul(x_t, self.W_i) + np.matmul(h_t, self.U_i) + self.b_i,
        )
        f_t = sigmoid_fn(
            np.matmul(x_t, self.W_f) + np.matmul(h_t, self.U_f) + self.b_f,
        )
        c_tilde = tanh_fn(
            np.matmul(x_t, self.W_c) + np.matmul(h_t, self.U_c) + self.b_c,
        )
        o_t = sigmoid_fn(
            np.matmul(x_t, self.W_o) + np.matmul(h_t, self.U_o) + self.b_o,
        )
        c_t = f_t * c_t + i_t * c_tilde
        h_t = o_t * tanh_fn(c_t)
        return c_t, h_t


if __name__ == "__main__":
    # data set shape = (num_samples, num_timesteps, num_features)
    # input shape = (num_timesteps, num_features)
    # If return_sequences is True:
    # output shape = (num_timesteps, units)
    # Else:
    # output shape = (1, units)
    x = np.random.normal(size=(1, 3, 2))
    units = 4
    return_sequences = True

    # num_features: 2
    # units: 4
    # h_t shape = (4),        (units)
    # W shape   = (2, 4),     (num_features, units)
    # U shape   = (4, 4),     (units, units)
    # b shape   = (4),        (units)
    #
    # matmul(x, W)      (1, 2)*(2,4) => (4)
    # matmul(h, U)      (1, 4)*(4,4) => (4)
    # intern + b        (4)+(4)   => (4)
    model = Sequential()
    model.add(
        LSTM(
            units=units,
            return_sequences=return_sequences,
            input_shape=x.shape[1:],
        ),
    )
    model.compile(
        loss="mse",
        optimizer="Adam",
    )
    # model.summary()

    rnn = LSTMInference(
        lstm_layer=model.layers[0],
        return_sequences=return_sequences,
    )
    output_rnn_own = rnn(x[0])  # 10.5
    print(output_rnn_own)
    print("\n\n")
    output_rnn_tf = model.predict(x[[0]])
    print(output_rnn_tf)  # 10.5
    if not np.all(np.isclose(output_rnn_own - output_rnn_tf, 0.0, atol=1e-06)):
        raise AssertionError
