import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = 32
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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / np.float32(255.0)
    x_test = x_test / np.float32(255.0)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train[:-10000], y_train[:-10000]))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_train[-10000:], y_train[-10000:]))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_dataset = prepare(train_dataset, shuffle=True, augment=True)
    validation_dataset = prepare(validation_dataset)
    test_dataset = prepare(test_dataset)

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = get_dataset()
