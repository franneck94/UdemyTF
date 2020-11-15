import os

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Sequential


DATA_DIR = os.path.abspath("C:/Users/Jan/Documents/DogsAndCats")
IMG_SIZE = 64
IMG_DEPTH = 3


class DOGSCATS:
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.33) -> None:
        # Helper variables
        self.num_classes = 2
        self.batch_size = 128
        number_validation_batches = int(len(self.train_dataset) * validation_size)
        # Load the data set
        self.train_dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory=DATA_DIR,
            validation_split=test_size,
            seed=0,
            label_mode="binary",
            batch_size=self.batch_size,
            image_size=(IMG_SIZE, IMG_SIZE),
            subset="training"
        )
        self.test_dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory=DATA_DIR,
            validation_split=test_size,
            seed=0,
            label_mode="binary",
            batch_size=self.batch_size,
            image_size=(IMG_SIZE, IMG_SIZE),
            subset="validation"
        )
        self.val_dataset = self.train_dataset.take(number_validation_batches)
        self.train_dataset = self.train_dataset.skip(number_validation_batches)
        # Dataset attributes
        self.train_size = len(self.train_dataset) * self.batch_size
        self.test_size = len(self.test_dataset) * self.batch_size
        self.val_size = len(self.val_dataset) * self.batch_size
        self.width = IMG_SIZE
        self.height = IMG_SIZE
        self.depth = IMG_DEPTH
        self.img_shape = (self.width, self.height, self.depth)
        # Prepare datasets
        self.train_dataset = self.train_dataset.cache()
        self.test_dataset = self.test_dataset.cache()
        self.val_dataset = self.val_dataset.cache()
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

        model.add(Rescaling(scale=(1.0 / 255.0), offset=0.0))

        return model

    @staticmethod
    def _build_data_augmentation() -> Sequential:
        model = Sequential()

        model.add(RandomRotation(factor=0.08))
        model.add(RandomTranslation(height_factor=0.08, width_factor=0.08))
        model.add(RandomZoom(height_factor=0.08, width_factor=0.08))

        return model

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False,
        augment: bool = False
    ) -> tf.data.Dataset:
        preprocessing_model = self._build_preprocessing()
        dataset = dataset.map(
            map_func=lambda x, y: (preprocessing_model(x, training=False), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        if augment:
            data_augmentation_model = self._build_data_augmentation()
            dataset = dataset.map(
                map_func=lambda x, y: (data_augmentation_model(x, training=False), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    data = DOGSCATS()
