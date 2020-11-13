from tensorflow.keras.datasets import mnist

from tf_utils.plotting import display_digit


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    for i in range(3):
        display_digit(x_train[i], label=y_train[i])
