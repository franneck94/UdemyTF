import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
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
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
    bias_initializer: tf.keras.initializers.Initializer,
    activation_str: str,
    dropout_rate: float,
    use_bn: bool,
    use_global_pooling: bool,
    use_additional_dense_layer: bool
) -> Model:
    # Input
    input_img = Input(shape=img_shape)
    # Conv Block 1
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(input_img)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 2
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Dense Part
    if use_global_pooling:
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    if use_additional_dense_layer:
        x = Dense(units=dense_layer_size)(x)
        if activation_str == "LeakyReLU":
            x = LeakyReLU()(x)
        else:
            x = Activation(activation_str)(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    # Build the model
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
    model.summary()
    return model


def schedule_fn(epoch: int) -> float:
    learning_rate = 1e-3
    if epoch < 5:
        learning_rate = 1e-3
    elif epoch < 20:
        learning_rate = 5e-4
    else:
        learning_rate = 1e-4
    return learning_rate


def schedule_fn2(epoch: int) -> float:
    if epoch < 10:
        return 1e-3
    else:
        return 1e-3 * np.exp(0.1 * (10 - epoch))


if __name__ == "__main__":
    data = DOGSCATS()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    # Global params
    epochs = 10
    batch_size = 256

    params = {
        "optimizer": Adam,
        "learning_rate": 0.001,
        "filter_block1": 32,
        "kernel_size_block1": 3,
        "filter_block2": 64,
        "kernel_size_block2": 3,
        "filter_block3": 128,
        "kernel_size_block3": 3,
        "dense_layer_size": 1024,
        # GlorotUniform, GlorotNormal, RandomNormal
        # RandomUniform, VarianceScaling
        "kernel_initializer": 'GlorotUniform',
        "bias_initializer": 'zeros',
        # relu, elu, LeakyReLU
        "activation_str": "relu",
        # 0.05, 0.1, 0.2
        "dropout_rate": 0.00,
        # True, False
        "use_bn": True,
        # True, False
        "use_global_pooling": True,
        # True, False
        "use_additional_dense_layer": True,
    }

    model = build_model(
        img_shape,
        num_classes,
        **params
    )

    lrs_callback = LearningRateScheduler(
        schedule=schedule_fn2,
        verbose=1
    )

    plateau_callback = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.95,
        patience=2,
        verbose=1,
        min_lr=1e-5
    )

    es_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    # model.fit(
    #     train_dataset,
    #     verbose=1,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     callbacks=[lrs_callback, plateau_callback, es_callback],
    #     validation_data=val_dataset,
    # )

    # model.save_weights(MODEL_FILE_PATH)
    model.load_weights(MODEL_FILE_PATH)

    images_path = os.path.abspath(CUSTOM_IMAGE_PATH)
    image_names = [f for f in os.listdir(images_path) if '.jpg' in f or '.jpeg' in f or '.png' in f]
    class_names = ["cat", "dog"]

    for image_name in image_names:
        image_path = os.path.join(images_path, image_name)
        x = data.load_and_preprocess_custom_image(image_path)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred_class = np.argmax(y_pred)
        y_pred_prob = y_pred[y_pred_class]
        plt.imshow(x)
        plt.title(f"Predicted class: {class_names[y_pred_class]}, Prob: {y_pred_prob}")
        plt.show()
