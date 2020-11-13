import numpy as np
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

from mnistDataValidation import MNIST


def model_fn() -> Model:
    input_img = Input(shape=x_train.shape[1:])

    x = Conv2D(filters=32, kernel_size=3, padding='same')(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
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

    # Build the model
    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    mnist = MNIST()
    mnist.data_augmentation(augment_size=5_000)
    mnist.data_preprocessing(preprocess_mode="MinMax")
    x_train, y_train = mnist.get_train_set()
    x_test, y_test = mnist.get_test_set()
    num_classes = mnist.num_classes

    # Model params
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    epochs = 3
    batch_size = 128
    keras_clf = KerasClassifier(
        build_fn=model_fn,
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
