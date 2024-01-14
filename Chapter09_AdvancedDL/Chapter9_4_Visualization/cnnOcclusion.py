import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import ReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import Optimizer

from tf_utils.callbacks import schedule_fn2
from tf_utils.dogsCatsDataAdvanced import DOGSCATS
from tf_utils.plotting import get_occlusion


np.random.seed(0)  # noqa: NPY002
tf.random.set_seed(0)


MODELS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/models/")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "dogs_cats.h5")
LOGS_DIR = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def build_model(
    img_shape: tuple[int, int, int],
    num_classes: int,
    optimizer: Optimizer,
    learning_rate: float,
    filter_block1: int,
    kernel_size_block1: int,
    filter_block2: int,
    kernel_size_block2: int,
    filter_block3: int,
    kernel_size_block3: int,
    dense_layer_size: int,
    activation_cls: Activation,
    dropout_rate: float,
    use_batch_normalization: bool,
    use_dense: bool,
    use_global_pooling: bool,
) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
    )(input_img)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    x = GlobalAveragePooling2D()(x) if use_global_pooling else Flatten()(x)
    if use_dense:
        x = Dense(
            units=dense_layer_size,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = activation_cls(x)
    x = Dense(
        units=num_classes,
    )(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    data_dir = os.path.join("C:/Users/Jan/Documents/DogsAndCats")
    data = DOGSCATS(data_dir=data_dir)

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    # Global params
    epochs = 10
    batch_size = 128

    params = {
        "dense_layer_size": 128,
        "optimizer": Adam,
        "learning_rate": 1e-3,
        "filter_block1": 32,
        "kernel_size_block1": 3,
        "filter_block2": 64,
        "kernel_size_block2": 3,
        "filter_block3": 128,
        "kernel_size_block3": 3,
        "activation_cls": ReLU(),
        "dropout_rate": 0.0,
        "use_batch_normalization": True,
        "use_dense": True,
        "use_global_pooling": True,
    }

    model = build_model(img_shape, num_classes, **params)

    lrs_callback = LearningRateScheduler(schedule=schedule_fn2, verbose=1)

    es_callback = EarlyStopping(
        monitor="val_accuracy",
        patience=30,
        verbose=1,
        restore_best_weights=True,
    )

    model.fit(
        train_dataset,
        verbose=1,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[lrs_callback, es_callback],
        validation_data=val_dataset,
    )

    model.save_weights(filepath=MODEL_FILE_PATH)
    model.load_weights(filepath=MODEL_FILE_PATH)

    score = model.evaluate(val_dataset, verbose=0, batch_size=batch_size)
    print(f"Test performance: {score}")

    data_tuple = test_dataset.take(1).as_numpy_iterator().next()
    img = data_tuple[0][0]
    label = data_tuple[1][0]

    get_occlusion(img=img, label=label, box_size=4, step_size=4, model=model)
