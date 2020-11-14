from typing import Tuple

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

from mnistDataAdvanced import IMG_SHAPE
from mnistDataAdvanced import NUM_CLASSES
from mnistDataAdvanced import get_dataset


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


def train_and_evaluate_model(model: Model) -> None:
    """Train and test the transfer learning model

    Parameters
    ----------
    model : Model
        The transfer learning model
    """
    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam",
        metrics=["accuracy"]
    )
    train_dataset, test_dataset = get_dataset()
    model.fit(
        train_dataset,
        epochs=20,
        validation_data=test_dataset
    )
    model.evaluate(
        test_dataset
    )


if __name__ == "__main__":
    model = build_model(IMG_SHAPE, NUM_CLASSES)

    train_and_evaluate_model(model)
