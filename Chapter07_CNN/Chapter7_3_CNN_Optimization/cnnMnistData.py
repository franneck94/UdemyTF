from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.models import Model

from Chapter07_CNN.Chapter7_3_CNN_Optimization.mnistData4 import MNIST


def build_model(img_shape: tuple[int, int, int], num_classes: int) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(filters=32, kernel_size=5, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=5, padding="same")(x)
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
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    model.summary()

    return model


if __name__ == "__main__":
    data = MNIST()
    x_train, y_train = data.get_train_set()
    x_test, y_test = data.get_test_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    model = build_model(img_shape, num_classes)

    model.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )

    model.fit(
        x=x_train / 255.0, y=y_train, epochs=3, validation_data=(x_test, y_test)
    )

    scores = model.evaluate(x=x_test / 255.0, y=y_test)

    print(f"Scores: {scores}")
