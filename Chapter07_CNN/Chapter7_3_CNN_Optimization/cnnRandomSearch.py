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
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

from Chapter07_CNN.Chapter7_3_CNN_Optimization.mnistData4 import MNIST


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

    param_distributions = {
        "filters_1": randint(8, 64),
        "kernel_size_1": randint(3, 8),
        "filters_2": randint(8, 64),
        "kernel_size_2": randint(3, 8),
        "filters_3": randint(8, 64),
        "kernel_size_3": randint(3, 8),
    }

    # Code has changed comapred to the videos: https://adriangb.com/scikeras/stable/migration.html
    keras_clf = KerasClassifier(
        build_fn=build_model, epochs=3, batch_size=128, verbose=1
    )

    rand_cv = RandomizedSearchCV(
        estimator=keras_clf,
        param_distributions=param_distributions,
        n_iter=10,
        n_jobs=1,
        verbose=0,
        cv=3,
    )

    rand_result = rand_cv.fit(X=x_train, y=y_train)

    print(f"Best: {rand_result.best_score_} using {rand_result.best_params_}")

    means = rand_result.cv_results_["mean_test_score"]
    stds = rand_result.cv_results_["std_test_score"]
    params = rand_result.cv_results_["params"]

    for mean, std, param in zip(means, stds, params):
        print(f"Acc: {mean} (+/- {std * 2}) with: {param}")
