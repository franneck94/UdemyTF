from typing import Tuple

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tf_utils.cifarDataAdvanced import CIFAR10


def build_model(
    img_shape: Tuple[int, int, int],
    num_classes: int
) -> Model:
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape,
        classes=num_classes
    )

    print(f"Number of layers in the base model: {len(base_model.layers)}")
    fine_tune_at = len(base_model.layers) - 10
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    input_img = Input(shape=img_shape)
    x = base_model(input_img, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=input_img,
        outputs=y_pred
    )

    model.summary()

    return model


if __name__ == "__main__":
    # Own model had accuracy of 0.85 on test set
    data = CIFAR10()

    train_dataset = data.get_train_set()
    test_dataset = data.get_test_set()
    val_dataset = data.get_val_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    model = build_model(img_shape, num_classes)

    optimizer = Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset
    )

    model.evaluate(
        test_dataset
    )
