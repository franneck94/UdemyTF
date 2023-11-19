import numpy as np
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


def get_dataset(
    num_features: int, num_classes: int
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)
    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)
    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_classes: int) -> Sequential:
    model = Sequential()
    model.add(Dense(units=256, input_shape=(num_features,)))
    model.add(Activation("relu"))
    model.add(Dense(units=128, input_shape=(num_features,)))
    model.add(Activation("relu"))
    model.add(Dense(units=64, input_shape=(num_features,)))
    model.add(Activation("relu"))
    model.add(Dense(units=num_classes))
    model.add(Activation("softmax"))
    model.summary()
    return model


def main() -> None:
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = get_dataset(
        num_features, num_classes
    )

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    model = build_model(num_features, num_classes)

    opt = Adam(learning_rate=0.002)

    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=15,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    scores = model.evaluate(x=x_test, y=y_test)

    print(f"Scores: {scores}")


if __name__ == "__main__":
    main()
