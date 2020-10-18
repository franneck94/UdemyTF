from typing import Tuple

import tensorflow as tf


BATCH_SIZE = 32
IMG_SIZE = 32
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
NUM_CLASSES = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_preprocessing() -> tf.keras.Sequential:
    """Build the preprocessing model, to rescale the images.

    Returns
    -------
    tf.keras.Sequential
        The preprocessing model
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.experimental.preprocessing.Rescaling(
            scale=1.0 / 127.5,
            offset=-1.0
        )
    )
    return model


def build_data_augmentation() -> tf.keras.Sequential:
    """Build the data augmentation model, to random rotate,
    zoom and translate the images.

    Returns
    -------
    tf.keras.Sequential
        The preprocessing model
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.experimental.preprocessing.RandomRotation(
            factor=0.0625  # 2 pixels
        )
    )
    model.add(
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=0.0625,  # 2 pixels
            width_factor=0.0625  # 2 pixels
        )
    )
    model.add(
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=0.0625,  # 2 pixels
            width_factor=0.0625  # 2 pixels
        )
    )
    return model


def prepare(
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
    # Resize and rescale all datasets
    preprocessing_model = build_preprocessing()
    dataset = dataset.map(
        map_func=lambda x, y: (preprocessing_model(x), y),
        num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # Batch all datasets
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    # Use data augmentation only on the training set
    if augment:
        data_augmentation = build_data_augmentation()
        dataset = dataset.map(
            map_func=lambda x, y: (data_augmentation(x), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Use buffered prefecting on all datasets
    return dataset.prefetch(buffer_size=AUTOTUNE)


def get_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generate the train, validation and test set

    Returns
    -------
    Tuple
        (train_dataset, validation_dataset, test_dataset)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(
        y=y_train,
        num_classes=NUM_CLASSES
    )
    y_test = tf.keras.utils.to_categorical(
        y=y_test,
        num_classes=NUM_CLASSES
    )

    validation_size = x_train.shape[0] // 5
    train_dataset = tf.data.Dataset.from_tensor_slices(
        tensors=(x_train[:-validation_size], y_train[:-validation_size])
    )
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        tensors=(x_train[-validation_size:], y_train[-validation_size:])
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        tensors=(x_test, y_test)
    )

    train_dataset = prepare(train_dataset, shuffle=True, augment=True)
    validation_dataset = prepare(validation_dataset)
    test_dataset = prepare(test_dataset)

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = get_dataset()
