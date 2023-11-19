# pyright: reportMissingImports=false
import os

import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical
from packaging import version
from skimage import transform
from sklearn.model_selection import train_test_split


required_version = version.parse("2.10")
installed_version = version.parse(".".join(tf.__version__.split(".")[:2]))
if installed_version > required_version:
    from keras.layers.experimental.preprocessing import RandomRotation
    from keras.layers.experimental.preprocessing import RandomTranslation
    from keras.layers.experimental.preprocessing import RandomZoom
else:
    from keras.layers.preprocessing.image_preprocessing import RandomRotation
    from keras.layers.preprocessing.image_preprocessing import RandomTranslation
    from keras.layers.preprocessing.image_preprocessing import RandomZoom

np.random.seed(0)
tf.random.set_seed(0)


class DOGSCATS:
    def __init__(
        self,
        data_dir: str,
        test_size: float = 0.2,
        validation_size: float = 0.33,
    ) -> None:
        # User-definen constants
        self.num_classes = 2
        self.batch_size = 128
        # Load the data set
        x_filepath = os.path.join(data_dir, "x.npy")
        y_filepath = os.path.join(data_dir, "y.npy")
        x = np.load(x_filepath)
        y = np.load(y_filepath)
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=validation_size,
        )
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y data
        self.y_train = to_categorical(
            y_train,
            num_classes=self.num_classes,
        )
        self.y_test = to_categorical(
            y_test,
            num_classes=self.num_classes,
        )
        self.y_val = to_categorical(
            y_val,
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
        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        )
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_val, self.y_val)
        )
        self.train_dataset = self._prepare_dataset(
            self.train_dataset,
            shuffle=True,
            augment=True,
        )
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    @staticmethod
    def load_and_preprocess_custom_image(
        image_file_path: str,
        img_shape: tuple[int, int, int],
    ) -> np.ndarray:
        img = cv2.imread(
            image_file_path,
            cv2.IMREAD_COLOR,
        )
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )
        return transform.resize(
            image=img,
            output_shape=img_shape,
        )

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
        augment: bool = False,
    ) -> tf.data.Dataset:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        dataset = dataset.batch(batch_size=self.batch_size)

        if augment:
            data_augmentation_model = self._build_data_augmentation()
            dataset = dataset.map(
                map_func=lambda x, y: (
                    data_augmentation_model(x, training=False),
                    y,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    data_dir = os.path.join("C:/Users/Jan/Documents/DogsAndCats")
    d = DOGSCATS(data_dir=data_dir)
