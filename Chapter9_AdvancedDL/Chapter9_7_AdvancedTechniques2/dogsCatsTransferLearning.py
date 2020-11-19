from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tf_utils.callbacks import schedule_fn2
from tf_utils.dogsCatsDataAdvanced import DOGSCATS


IMAGENET_SIZE = 224
IMAGENET_DEPTH = 3
IMAGENET_SHAPE = (IMAGENET_SIZE, IMAGENET_SIZE, IMAGENET_DEPTH)


def build_model(img_shape, num_classes) -> Model:
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGENET_SHAPE
    )

    print(f"Number of layers in the base model: {len(base_model.layers)}")
    fine_tune_at = len(base_model.layers) - 1
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    input_img = Input(shape=img_shape)
    x = Rescaling(scale=2.0, offset=-1.0)(input_img)
    x = Resizing(height=IMAGENET_SIZE, width=IMAGENET_SIZE)(x)
    x = base_model(x)
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
    """
    Best model from chapter 9_3: 0.9034 accuracy
    Best model from here:        0.9559 accuracy
    """
    data = DOGSCATS()

    train_dataset = data.get_train_set()
    test_dataset = data.get_test_set()
    val_dataset = data.get_val_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    model = build_model(img_shape, num_classes)

    optimizer = Adam(learning_rate=5e-4)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    epochs = 100

    lrs_callback = LearningRateScheduler(
        schedule=schedule_fn2,
        verbose=1
    )

    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=1,
        restore_best_weights=True,
        min_delta=0.0005
    )

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        verbose=1,
        callbacks=[lrs_callback, es_callback]
    )

    scores = model.evaluate(
        test_dataset,
        verbose=0
    )
    print(f"Scores: {scores}")
