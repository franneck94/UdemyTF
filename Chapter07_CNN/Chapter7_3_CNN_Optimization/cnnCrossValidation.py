import numpy as np
import tensorflow as tf
from keras.layers import Activation, Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.models import Model
from mnistData import MNIST
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(0)
tf.random.set_seed(0)


def build_model() -> Model:
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(filters=16, kernel_size=3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64, kernel_size=5, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=5, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=10)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    model.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    data = MNIST(with_normalization=True)

    x_train, y_train = data.get_train_set()

    keras_clf = KerasClassifier(
        build_fn=build_model, epochs=10, batch_size=128, verbose=1
    )

    scores = cross_val_score(
        estimator=keras_clf, X=x_train, y=y_train, cv=3, n_jobs=1
    )

    print(f"Score list: {scores}")
    mean_scores = np.mean(scores)
    std_scores = np.std(scores)
    print(f"Mean Acc: {mean_scores:.6} (+/- {std_scores:0.6})")
