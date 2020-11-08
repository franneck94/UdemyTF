import tensorflow as tf


def my_adder(a, b):
    return a + b


def my_adder2(a: int, b: int) -> int:
    return a + b


def print_model(model: tf.keras.models.Sequential) -> None:
    model.summary()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

print_model(model)
