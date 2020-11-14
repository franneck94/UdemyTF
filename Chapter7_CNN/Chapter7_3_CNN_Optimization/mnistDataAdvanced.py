from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


BATCH_SIZE = 128
IMG_SIZE = 28
IMG_DEPTH = 1
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
NUM_CLASSES = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_preprocessing() -> Model:
    """Build the preprocessing model, to resize and rescale the images.

    Returns
    -------
    Model
        The preprocessing model
    """
    input_img = Input(shape=IMG_SHAPE)
    preprocessed_img = Rescaling(scale=1.0 / 255.0, offset=0.0)(input_img)

    model = Model(
        inputs=[input_img],
        outputs=[preprocessed_img]
    )
    model.summary()

    return model


def build_data_augmentation() -> Model:
    """Build the data augmentation model, to random rotate,
    zoom and translate the images.

    Returns
    -------
    Model
        The preprocessing model
    """
    input_img = Input(shape=IMG_SHAPE)
    x = RandomRotation(factor=0.05)(input_img)
    x = RandomZoom(height_factor=0.05, width_factor=0.05)(x)
    augmented_img = RandomTranslation(height_factor=0.05, width_factor=0.05)(x)

    model = Model(
        inputs=[input_img],
        outputs=[augmented_img]
    )
    model.summary()

    return model


def prepare_dataset(
    dataset: tf.data.Dataset,
    shuffle: bool = False,
    augment: bool = False
) -> tf.data.Dataset:
    """Prepare the dataset object with preprocessing and data augmentation.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset object
    shuffle : bool, optional
        Whether to shuffle the dataset, by default False
    augment : bool, optional
        Whether to augment the train dataset, by default False

    Returns
    -------
    tf.data.Dataset
        The prepared dataset
    """
    preprocessing_model = build_preprocessing()
    dataset = dataset.map(
        map_func=lambda x, y: (preprocessing_model(x, training=False), y),
        num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1_000)

    dataset = dataset.batch(batch_size=BATCH_SIZE)

    if augment:
        data_augmentation_model = build_data_augmentation()
        dataset = dataset.map(
            map_func=lambda x, y: (data_augmentation_model(x, training=False), y),
            num_parallel_calls=AUTOTUNE
        )

    return dataset.prefetch(buffer_size=AUTOTUNE)


def get_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Generate the train, validation and test set

    Returns
    -------
    Tuple
        (train_dataset, test_dataset)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_train = np.reshape(x_train, (-1, IMG_SIZE, IMG_SIZE, 1))
    x_test = x_test.astype(np.float32)
    x_test = np.reshape(x_test, (-1, IMG_SIZE, IMG_SIZE, 1))

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES, dtype="float32")
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES, dtype="float32")

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    print(train_dataset)

    train_dataset = prepare_dataset(train_dataset, shuffle=True, augment=True)
    test_dataset = prepare_dataset(test_dataset)

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset()
