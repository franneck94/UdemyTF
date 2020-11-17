import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_batch_ops import batch

from tf_utils.callbacks import LRTensorBoard
from tf_utils.callbacks import schedule_fn
from tf_utils.callbacks import schedule_fn2
from tf_utils.callbacks import schedule_fn3
from tf_utils.dogsCatsDataAdvanced import DOGSCATS


np.random.seed(0)
tf.random.set_seed(0)


CUSTOM_IMAGE_PATH = os.path.abspath("C:/Users/Jan/Documents/DogsAndCats/custom/")
MODELS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/models/")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "dogs_cats.h5")
LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def build_model(
    img_shape: Tuple[int, int, int],
    num_classes: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float,
    filter_block1: int,
    kernel_size_block1: int,
    filter_block2: int,
    kernel_size_block2: int,
    filter_block3: int,
    kernel_size_block3: int,
    dense_layer_size: int,
    kernel_initializer: tf.keras.initializers.Initializer,
    activation_cls: tf.keras.layers.Activation,
    dropout_rate: float,
    use_batch_normalization: bool
) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
        kernel_initializer=kernel_initializer
    )(input_img)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding="same",
        kernel_initializer=kernel_initializer
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
        kernel_initializer=kernel_initializer
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding="same",
        kernel_initializer=kernel_initializer
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
        kernel_initializer=kernel_initializer
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = activation_cls(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding="same",
        kernel_initializer=kernel_initializer
    )(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)
    x = activation_cls(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(
        units=num_classes,
        kernel_initializer=kernel_initializer
    )(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    data = DOGSCATS()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    # Global params
    epochs = 40
    batch_size = 128

    params = {
        "optimizer": Adam,
        "learning_rate": 1e-3,
        "filter_block1": 32,
        "kernel_size_block1": 3,
        "filter_block2": 64,
        "kernel_size_block2": 3,
        "filter_block3": 128,
        "kernel_size_block3": 3,
        "dense_layer_size": 128,
        "kernel_initializer": "GlorotUniform",
        "activation_cls": ReLU(),
        "dropout_rate": 0.00,
        "use_batch_normalization": False,
    }

    model = build_model(
        img_shape,
        num_classes,
        **params
    )

    model.fit(
        train_dataset,
        verbose=1,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_dataset,
    )

    model.save_weights(MODEL_FILE_PATH)
    model.load_weights(MODEL_FILE_PATH)

    images_path = os.path.abspath(CUSTOM_IMAGE_PATH)
    image_names = [f for f in os.listdir(images_path) if ".jpg" in f or ".jpeg" in f or ".png" in f]
    class_names = ["cat", "dog"]

    # for image_name in image_names:
    #     image_path = os.path.join(images_path, image_name)
    #     x = data.load_and_preprocess_custom_image(image_path)
    #     y_pred = model.predict(np.expand_dims(x, axis=0))[0]
    #     y_pred_class = np.argmax(y_pred)
    #     y_pred_prob = y_pred[y_pred_class]
    #     plt.imshow(x)
    #     plt.title(f"Predicted class: {class_names[y_pred_class]}, Prob: {y_pred_prob}")
    #     plt.show()
