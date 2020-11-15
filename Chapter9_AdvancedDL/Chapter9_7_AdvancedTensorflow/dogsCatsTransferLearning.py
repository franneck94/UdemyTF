from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tf_utils.dogsCatsDataAdvanced import DOGSCATS
from tf_utils.dogsCatsDataAdvanced import IMG_SHAPE
from tf_utils.dogsCatsDataAdvanced import NUM_TARGETS


def build_model() -> Model:
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SHAPE,
        classes=NUM_TARGETS
    )

    fine_tune_at_layer_idx = len(base_model.layers) - 10
    for layer in base_model.layers[:fine_tune_at_layer_idx]:
        layer.trainable = False

    input_img = Input(shape=IMG_SHAPE)
    x = base_model(input_img)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=NUM_TARGETS)(x)
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

    model = build_model()

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
