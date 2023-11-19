import os

import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


LOGS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def get_model_log_dir(model_name: str, model_run: int) -> str:
    return os.path.join(LOGS_DIR, f"{model_name}_{model_run}")


def prepare_dataset(num_features: int, num_targets: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_targets, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_targets, dtype=np.float32)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def build_model(
    num_features: int, num_targets: int, num_extra_layers: int = 2
) -> tuple[Sequential, TensorBoard]:
    log_name = get_model_log_dir("mnist", num_extra_layers)
    model = Sequential()
    model.add(Dense(units=500, input_shape=(num_features,)))
    model.add(Activation("relu"))
    for _ in range(num_extra_layers):
        model.add(Dense(units=250))
        model.add(Activation("relu"))
    model.add(Dense(units=num_targets))
    model.add(Activation("softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )
    tb_callback = TensorBoard(
        log_dir=log_name, histogram_freq=0, write_graph=False
    )
    return model, tb_callback


def main() -> None:
    num_features = 784
    num_targets = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(
        num_features, num_targets
    )

    for num_extra_layers in range(1, 4):
        model, tb_callback = build_model(
            num_features, num_targets, num_extra_layers
        )

        model.fit(
            x=x_train,
            y=y_train,
            epochs=20,
            batch_size=128,
            verbose=2,
            validation_data=(x_test, y_test),
            callbacks=[tb_callback],
        )

        scores = model.evaluate(x=x_test, y=y_test, verbose=0)
        print(f"Scores before saving: {scores}")


if __name__ == "__main__":
    main()
