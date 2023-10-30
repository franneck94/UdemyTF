# pyright: reportMissingImports=false
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from packaging import version

from tf_utils.callbacks import schedule_fn2
from tf_utils.cifarDataAdvanced import CIFAR10


required_version = version.parse("2.10")
installed_version = version.parse(".".join(tf.__version__.split(".")[:2]))
if installed_version > required_version:
    from keras.layers.experimental.preprocessing import Rescaling
    from keras.layers.experimental.preprocessing import Resizing
else:
    from keras.layers.preprocessing.image_preprocessing import Rescaling
    from keras.layers.preprocessing.image_preprocessing import Resizing


IMAGENET_SIZE = 96
IMAGENET_DEPTH = 3
IMAGENET_SHAPE = (IMAGENET_SIZE, IMAGENET_SIZE, IMAGENET_DEPTH)


def build_model(img_shape: tuple, num_classes: int) -> Model:
    base_model = MobileNetV2(
        include_top=False, weights="imagenet", input_shape=IMAGENET_SHAPE
    )

    num_layers = len(base_model.layers)
    print(f"Number of layers in the base model: {num_layers}")
    fine_tune_at = num_layers - 10
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    input_img = Input(shape=img_shape)
    x = Rescaling(scale=2.0, offset=-1.0)(input_img)
    x = Resizing(height=IMAGENET_SIZE, width=IMAGENET_SIZE)(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    model.summary()

    return model


if __name__ == "__main__":
    """
    Best model from chapter 8:   0.7200 accuracy
    Best model from chapter 9_3: 0.8361 accuracy
    Best model from chapter 9_7: 0.8993 accuracy
    """
    data = CIFAR10()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    # Global params
    epochs = 100

    model = build_model(img_shape, num_classes)

    opt = Adam(learning_rate=5e-4)

    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    lrs_callback = LearningRateScheduler(schedule=schedule_fn2, verbose=1)

    es_callback = EarlyStopping(
        monitor="val_loss", patience=30, verbose=1, restore_best_weights=True
    )

    model.fit(
        train_dataset,
        verbose=1,
        epochs=epochs,
        callbacks=[lrs_callback, es_callback],
        validation_data=val_dataset,
    )

    scores = model.evaluate(val_dataset, verbose=0)
    print(f"Scores: {scores}")
