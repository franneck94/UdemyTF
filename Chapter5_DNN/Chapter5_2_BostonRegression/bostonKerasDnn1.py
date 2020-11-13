import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def build_model(num_features: int, num_targets: int) -> Sequential:
    init_w = RandomUniform(minval=-1.0, maxval=1.0)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(Dense(units=16, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
    model.add(Activation("relu"))
    model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b))
    model.summary()

    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_train = np.reshape(y_train, newshape=(-1, 1))
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    y_test = np.reshape(y_test, newshape=(-1, 1))

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    model = build_model(num_features, num_targets)
