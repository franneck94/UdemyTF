# ruff: noqa: F401
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from packaging import version
from sklearn.model_selection import train_test_split


required_version = version.parse("2.10")
installed_version = version.parse(".".join(tf.__version__.split(".")[:2]))
if installed_version > required_version:
    from keras.layers.experimental.preprocessing import RandomRotation
    from keras.layers.experimental.preprocessing import RandomTranslation
    from keras.layers.experimental.preprocessing import RandomZoom
    from keras.layers.experimental.preprocessing import Rescaling
else:
    from keras.layers.preprocessing.image_preprocessing import RandomRotation
    from keras.layers.preprocessing.image_preprocessing import RandomTranslation
    from keras.layers.preprocessing.image_preprocessing import RandomZoom
    from keras.layers.preprocessing.image_preprocessing import Rescaling


class MNIST:
    def __init__(
        self,
        validation_size: float = 0.33,
    ) -> None:
        # User-definen constants
        self.num_classes = 10
        self.batch_size = 128
        # Load the data set
        (
            (self.x_train, self.y_train),
            (
                self.x_test,
                self.y_test,
            ),
        ) = mnist.load_data()
        # Split the dataset
        (self.x_train, self.x_val, self.y_train, self.y_val) = train_test_split(
            self.x_train,
            self.y_train,
            test_size=validation_size,
        )
        # Preprocess x data
        self.x_train = np.expand_dims(
            self.x_train,
            axis=-1,
        ).astype(np.float32)
        self.x_test = np.expand_dims(
            self.x_test,
            axis=-1,
        ).astype(np.float32)
        self.x_val = np.expand_dims(
            self.x_val,
            axis=-1,
        ).astype(np.float32)
        # Preprocess y data
        self.y_train = to_categorical(
            self.y_train,
            num_classes=self.num_classes,
        )
        self.y_test = to_categorical(
            self.y_test,
            num_classes=self.num_classes,
        )
        self.y_val = to_categorical(
            self.y_val,
            num_classes=self.num_classes,
        )
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)

    def get_train_set(self) -> tf.data.Dataset:
        pass

    def get_test_set(self) -> tf.data.Dataset:
        pass

    def get_val_set(self) -> tf.data.Dataset:
        pass
