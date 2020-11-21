import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tf_utils.caliHousingDataAdvanced import CALIHOUSING


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    error = tf.math.subtract(y_true, y_pred)
    squared_error = tf.math.square(error)
    numerator = tf.math.reduce_sum(squared_error)
    y_true_mean = tf.math.reduce_mean(y_true)
    mean_deviation = tf.math.subtract(y_true, y_true_mean)
    squared_mean_deviation = tf.math.square(mean_deviation)
    denominator = tf.reduce_sum(squared_mean_deviation)
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


def build_model(num_features: int, num_targets: int) -> Sequential:
    model = Sequential()

    model.add(Dense(units=16, input_shape=(num_features,)))
    model.add(Activation("relu"))
    model.add(Dense(units=64))
    model.add(Activation("relu"))
    model.add(Dense(units=16))
    model.add(Activation("relu"))
    model.add(Dense(units=num_targets))

    model.summary()
    return model


if __name__ == "__main__":
    """
    LinReg: ~0.6
    DNN: ~0.778
    """
    data = CALIHOUSING()

    train_dataset = data.get_train_set()
    test_dataset = data.get_test_set()
    val_dataset = data.get_val_set()

    num_features = data.num_features
    num_targets = data.num_targets

    # Model params
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    epochs = 200
    batch_size = 256

    model = build_model(num_features, num_targets)

    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=[r_squared]
    )

    model.fit(
        train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_dataset
    )

    score = model.evaluate(
        test_dataset,
        verbose=0
    )
    print(f"Score: {score}")
