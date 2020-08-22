import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 32
IMG_SIZE = 160
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocessing_fn():
    return tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 127.5, offset=-1)
    ])


def data_augmentation_fn():
    return tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.05),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.05)
    ])


def prepare(dataset, shuffle=False, augment=False):
    # Resize and rescale all datasets
    preprocessing = preprocessing_fn()
    dataset = dataset.map(lambda x, y: (preprocessing(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # Batch all datasets
    dataset = dataset.batch(BATCH_SIZE)

    # Use data augmentation only on the training set
    if augment:
        data_augmentation = data_augmentation_fn()
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Use buffered prefecting on all datasets
    return dataset.prefetch(buffer_size=AUTOTUNE)


def get_dataset():
    train_dataset, validation_dataset, test_dataset = tfds.load(
        "cats_vs_dogs",
        split=["train[:60%]", "train[60%:80%]", "train[80%:]"],
        as_supervised=True)

    train_dataset = prepare(train_dataset, shuffle=True, augment=True)
    validation_dataset = prepare(validation_dataset)
    test_dataset = prepare(test_dataset)

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = get_dataset()
