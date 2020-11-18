from typing import Tuple

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tf_utils.dogsCatsDataAdvanced import DOGSCATS


def build_model(
    img_shape: Tuple[int, int, int],
    num_classes: int
) -> Model:
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        classes=num_classes
    )

    fine_tune_at_layer_idx = len(base_model.layers) - 10
    for layer in base_model.layers[:fine_tune_at_layer_idx]:
        layer.trainable = False

    input_img = Input(shape=img_shape)
    x = base_model(input_img)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("sigmoid")(x)

    model = Model(
        inputs=input_img,
        outputs=y_pred
    )

    model.summary()

    return model


if __name__ == "__main__":
    data = DOGSCATS()

    train_dataset = data.get_train_set()
    test_dataset = data.get_test_set()
    val_dataset = data.get_val_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    model = build_model(img_shape, num_classes)

    optimizer = Adam(learning_rate=1e-5)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["binary_accuracy"]
    )

    model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset
    )

    model.evaluate(
        test_dataset
    )
