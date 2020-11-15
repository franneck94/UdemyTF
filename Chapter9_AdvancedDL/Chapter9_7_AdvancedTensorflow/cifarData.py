import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Sequential

import tensorflow_datasets as tfds


IMG_SIZE = 32
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
NUM_CLASSES = 10
BATCH_SIZE = 128


class Cifar10Data:
    def __init__(self) -> None:
        # User-definen constants
        self.num_classes = NUM_CLASSES
        self.batch_size = BATCH_SIZE
        # Load the data set
        self.train_dataset, self.val_dataset, self.test_dataset = tfds.load(
            name="cifar10",
            split=["train[:60%]", "train[60%:80%]", "train[80%:]"],
            as_supervised=True
        )
        # Dataset attributes
        self.train_size = len(self.train_dataset)
        self.test_size = len(self.test_dataset)
        self.val_size = len(self.val_dataset)
        self.width = IMG_SIZE
        self.height = IMG_SIZE
        self.depth = IMG_DEPTH
        self.img_shape = (self.width, self.height, self.depth)
        # tf.data Datasets
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True, augment=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    @staticmethod
    def _build_preprocessing() -> Sequential:
        model = Sequential()

        model.add(Resizing(height=IMG_SIZE, width=IMG_SIZE))
        model.add(Rescaling(scale=(1.0 / 127.5), offset=-1.0))

        return model

    @staticmethod
    def _build_data_augmentation() -> Sequential:
        model = Sequential()

        model.add(RandomRotation(factor=0.0625))
        model.add(RandomTranslation(height_factor=0.0625, width_factor=0.0625))
        model.add(RandomZoom(height_factor=0.0625, width_factor=0.0625))

        return model

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False,
        augment: bool = False
    ) -> tf.data.Dataset:
        preprocessing_model = self._build_preprocessing()
        dataset = dataset.map(
            map_func=lambda x, y: (preprocessing_model(x, training=False), tf.one_hot(y, depth=self.num_classes)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        dataset = dataset.batch(batch_size=self.batch_size)

        if augment:
            data_augmentation_model = self._build_data_augmentation()
            dataset = dataset.map(
                map_func=lambda x, y: (data_augmentation_model(x, training=False), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    data = Cifar10Data()
