from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from mnistDataAdvanced import MNIST


np.random.seed(0)
tf.random.set_seed(0)


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(filters=32, kernel_size=5, padding='same')(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )

    model.summary()

    return model


if __name__ == "__main__":
    data = MNIST()
    x_train, y_train = data.get_train_set()
    x_test, y_test = data.get_test_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    # Model params
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    epochs = 3
    batch_size = 128
    keras_clf = KerasClassifier(
        build_fn=build_model,
        img_shape=img_shape,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    scores = cross_val_score(
        estimator=keras_clf,
        X=x_train,
        y=y_train,
        cv=3,
        n_jobs=1
    )

    print(f"Score list: {scores}")
    mean_scores = np.mean(scores)
    std_scores = np.std(scores)
    print(f"Mean Acc: {mean_scores:.6} (+/- {std_scores:0.6})")
