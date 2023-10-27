import numpy as np
import tensorflow as tf
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.models import Model
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

from mnistData import MNIST


np.random.seed(0)
tf.random.set_seed(0)


def build_model(
    filters_1: int,
    kernel_size_1: int,
    filters_2: int,
    kernel_size_2: int,
    filters_3: int,
    kernel_size_3: int,
) -> Model:
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(filters=filters_1, kernel_size=kernel_size_1, padding="same")(
        input_img
    )
    x = Activation("relu")(x)
    x = Conv2D(filters=filters_1, kernel_size=kernel_size_1, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=filters_2, kernel_size=kernel_size_2, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filters_2, kernel_size=kernel_size_2, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=filters_3, kernel_size=kernel_size_3, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filters_3, kernel_size=kernel_size_3, padding="same")(x)
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

    param_grid = {
        "filters_1": [16, 32],
        "kernel_size_1": [3, 5],
        "filters_2": [32, 64],
        "kernel_size_2": [3, 5],
        "filters_3": [64, 128],
        "kernel_size_3": [5],
    }

    keras_clf = KerasClassifier(
        build_fn=build_model,
        epochs=3,
        batch_size=128,
        verbose=1,
        filters_1=16,
        filters_2=32,
        filters_3=64,
        kernel_size_1=3,
        kernel_size_2=3,
        kernel_size_3=5,
    )

    grid_cv = GridSearchCV(
        estimator=keras_clf, param_grid=param_grid, n_jobs=1, verbose=0, cv=3
    )

    grid_result = grid_cv.fit(x_train, y_train)

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]

    for mean, std, param in zip(means, stds, params):
        print(f"Acc: {mean} (+/- {std * 2}) with: {param}")
