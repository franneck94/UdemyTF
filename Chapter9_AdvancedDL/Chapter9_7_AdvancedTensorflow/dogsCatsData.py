from typing import Tuple

import tensorflow as tf

import tensorflow_datasets as tfds


BATCH_SIZE = 32
IMG_SIZE = 160
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
NUM_OUTPUTS = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_preprocessing() -> tf.keras.Sequential:
    """Build the preprocessing model, to resize and rescale the images.

    Returns
    -------
    tf.keras.Sequential
        The preprocessing model
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.experimental.preprocessing.Resizing(
            height=IMG_SIZE,
            width=IMG_SIZE
        )
    )
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
            factor=0.0625  # 10 pixel
        )
    )
    model.add(
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=0.0625,  # 10 pixel
            width_factor=0.0625  # 10 pixel
        )
    )
    model.add(
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=0.0625,  # 10 pixel
            width_factor=0.0625  # 10 pixel
        )
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
        map_func=lambda x, y: (preprocessing_model(x), y),
        num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1_000)

    dataset = dataset.batch(batch_size=BATCH_SIZE)

    if augment:
        data_augmentation_model = build_data_augmentation()
        dataset = dataset.map(
            map_func=lambda x, y: (data_augmentation_model(x), y),
            num_parallel_calls=AUTOTUNE
        )

    return dataset.prefetch(buffer_size=AUTOTUNE)


def get_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Generate the train, validation and test set

    Returns
    -------
    Tuple
        (train_dataset, validation_dataset, test_dataset)
    """
    train_dataset, validation_dataset, test_dataset = tfds.load(
        name="cats_vs_dogs",
        split=["train[:60%]", "train[60%:80%]", "train[80%:]"],
        as_supervised=True
    )

    train_dataset = prepare_dataset(train_dataset, shuffle=True, augment=True)
    validation_dataset = prepare_dataset(validation_dataset)
    test_dataset = prepare_dataset(test_dataset)

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = get_dataset()
