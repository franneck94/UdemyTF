import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.initializers import Constant, RandomUniform
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential


# Dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
y_train = y_train.reshape(-1, 1)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
y_test = y_test.reshape(-1, 1)

# Dataset variables
num_features = x_train.shape[1]
num_targets = y_train.shape[1]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# Model variables
hidden_layer_size = 50
nodes = [num_targets, hidden_layer_size, num_targets]  # input, hidden, output
epochs = 2200


def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


init_w = RandomUniform(minval=-1.0, maxval=1.0)
init_b = Constant(value=0.0)

# Define the DNN
model = Sequential()
model.add(
    Dense(
        units=16,
        kernel_initializer=init_w,
        bias_initializer=init_b,
        input_shape=(num_features,),
    )
)
model.add(Activation("relu"))
model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b))
model.summary()

model.fit(x_train, y_train, epochs=epochs)
model.evaluate(x_test, y_test)
