import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Wiki Link: https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    res_sum_squares = tf.math.reduce_sum(tf.math.square(y_true - y_pred))
    y_mean = tf.math.reduce_mean(y_true)
    total_sum_squares = tf.reduce_sum(tf.math.square(y_true - y_mean))
    r2 = 1.0 - tf.math.divide(res_sum_squares, total_sum_squares)
    return tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)


def get_dataset() -> (
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
):
    dataset = load_diabetes()
    x: np.ndarray = dataset.data
    y: np.ndarray = dataset.target.reshape(-1, 1)
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.3)
    x_train: np.ndarray = x_train_.astype(np.float32)
    x_test: np.ndarray = x_test_.astype(np.float32)
    y_train: np.ndarray = y_train_.astype(np.float32)
    y_test: np.ndarray = y_test_.astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_dataset()

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")


if __name__ == "__main__":
    main()
