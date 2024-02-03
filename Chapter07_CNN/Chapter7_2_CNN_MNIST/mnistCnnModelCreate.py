import numpy as np
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


def prepare_dataset(num_classes: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = x_test.astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(
    img_shape: tuple[int, int, int],
    num_classes: int,
) -> Sequential:
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=3, input_shape=img_shape))
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=32, kernel_size=3))
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(units=num_classes))
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_classes)

    model = build_model(img_shape, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    scores = model.evaluate(x=x_test, y=y_test, verbose=0)
    print(scores)
