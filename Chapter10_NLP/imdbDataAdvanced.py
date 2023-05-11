import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from keras.models import Sequential

import tensorflow_datasets as tfds
from tensorcross.utils import dataset_split


np.random.seed(0)
tf.random.set_seed(0)


class IMDB:
    def __init__(self, validation_size: float = 0.33) -> None:
        # User-definen constants
        self.num_classes = 2
        self.batch_size = 128
        # Load the data set
        dataset = tfds.load("imdb_reviews", as_supervised=True)
        self.train_dataset, self.val_dataset = dataset_split(
            dataset["train"], split_fraction=validation_size
        )
        self.test_dataset = dataset["test"]
        # Dataset attributes
        self.train_size = len(self.train_dataset)
        self.test_size = len(self.test_dataset)
        self.val_size = len(self.val_dataset)
        # Prepare Datasets
        # self.train_dataset = self._prepare_dataset(self.train_dataset)
        # self.test_dataset = self._prepare_dataset(self.test_dataset)
        # self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    def _build_preprocessing(self) -> Sequential:
        raise NotImplementedError

    def _prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        preprocessing_model = self._build_preprocessing()
        dataset = dataset.map(
            map_func=lambda x, y: (preprocessing_model(x, training=False), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    imdb_data = IMDB()
    train_dataset = imdb_data.get_train_set()
    for text_batch, label_batch in train_dataset.take(1):
        print(text_batch.numpy(), label_batch.numpy())
