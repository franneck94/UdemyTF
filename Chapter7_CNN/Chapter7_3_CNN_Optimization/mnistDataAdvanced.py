import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class MNIST:
    def __init__(self):
        # Load the data set
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # Convert to float32
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # Reshape the x data to shape (num_examples, width, height, depth)
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        # Save important data attributes as variables
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        self.num_classes = 10
        # Reshape the y data to one hot encoding
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        # Ndarray to Dataset object
        self.batch_size = 128
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True, augment=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    @staticmethod
    def _build_preprocessing() -> Sequential:
        """Build the preprocessing model, to resize and rescale the images.

        Returns
        -------
        Sequential
            The preprocessing model
        """
        model = Sequential()

        model.add(Rescaling(scale=1.0 / 255.0, offset=0.0))

        return model

    @staticmethod
    def _build_data_augmentation() -> Sequential:
        """Build the data augmentation model, to random rotate,
        zoom and translate the images.

        Returns
        -------
        Sequential
            The preprocessing model
        """
        model = Sequential()

        model.add(RandomRotation(factor=0.05))
        model.add(RandomZoom(height_factor=0.05, width_factor=0.05))
        model.add(RandomTranslation(height_factor=0.05, width_factor=0.05))

        return model

    def _prepare_dataset(
        self,
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
        preprocessing_model = self._build_preprocessing()
        dataset = dataset.map(
            map_func=lambda x, y: (preprocessing_model(x, training=False), y),
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
    data = MNIST()
