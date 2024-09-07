import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import TextVectorization
from keras.models import Sequential
from tensorcross.utils import dataset_split


np.random.seed(0)
tf.random.set_seed(0)


class IMDB:
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        validation_size: float = 0.33,
    ) -> None:
        # User-definen constants
        self.num_classes = 2
        self.batch_size = 128
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        # Load the data set
        dataset = tfds.load("imdb_reviews", as_supervised=True)
        self.train_dataset, self.val_dataset = dataset_split(
            dataset["train"],
            split_fraction=validation_size,
        )
        self.test_dataset = dataset["test"]
        # Dataset attributes
        self.train_size = len(self.train_dataset)
        self.test_size = len(self.test_dataset)
        self.val_size = len(self.val_dataset)
        # Vectorizaion layer
        self.vectorization_layer = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )
        text_data = self.train_dataset.map(lambda x, _: x)
        self.vectorization_layer.adapt(text_data)
        self.vocabulary = self.vectorization_layer.get_vocabulary()
        self.word_index = dict(
            zip(self.vocabulary, range(len(self.vocabulary))),
        )
        # Prepare Datasets
        self.train_dataset = self._prepare_dataset(self.train_dataset)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    def _build_preprocessing(self) -> Sequential:
        model = Sequential()

        model.add(self.vectorization_layer)

        return model

    def _mask_to_categorical(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        y = tf.one_hot(tf.cast(y, tf.int32), depth=self.num_classes)
        y = tf.cast(y, tf.float32)
        return x, y

    def _prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        preprocessing_model = self._build_preprocessing()

        dataset = dataset.map(
            map_func=lambda x, y: (preprocessing_model(x, training=False), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.map(
            map_func=self._mask_to_categorical,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    vocab_size = 20_000
    sequnce_length = 80
    imdb_data = IMDB(vocab_size, sequnce_length)
    train_dataset = imdb_data.get_train_set()
    for text_batch, label_batch in train_dataset.take(1):
        for i in range(3):
            print(text_batch[i].numpy(), label_batch[i].numpy())
