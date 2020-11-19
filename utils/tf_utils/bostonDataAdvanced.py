import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers.experimental.preprocessing import Normalization


np.random.seed(0)
tf.random.set_seed(0)


class BOSTON:
    def __init__(self, validation_size: float = 0.33) -> None:
        # User-definen constants
        self.num_targets = 1
        self.batch_size = 128
        # Load the data set
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        # Split the dataset
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y data
        self.y_train = np.reshape(y_train, (-1, self.num_targets)).astype(np.float32)
        self.y_test = np.reshape(y_test, (-1, self.num_targets)).astype(np.float32)
        self.y_val = np.reshape(y_val, (-1, self.num_targets)).astype(np.float32)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.num_features = self.x_train.shape[1]
        self.num_targets = self.y_train.shape[1]
        # Normalization variables
        self.normalization_layer = Normalization()
        self.normalization_layer.adapt(self.x_train)
        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        # Dataset preparation
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False
    ) -> tf.data.Dataset:
        dataset = dataset.map(
            map_func=lambda x, y: (
                tf.reshape(
                    self.normalization_layer(
                        tf.reshape(x, shape=(1, self.num_features)), training=False
                    ),
                    shape=(self.num_features,)
                ), y
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
