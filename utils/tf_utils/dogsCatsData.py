import os

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from skimage import transform
from sklearn.model_selection import train_test_split


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
            x, y, test_size=test_size
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=validation_size
        )
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)
        self.y_val = to_categorical(y_val, num_classes=self.num_classes)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)

    def get_train_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_val_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_val, self.y_val

    @staticmethod
    def load_and_preprocess_custom_image(
        image_file_path: str,
        img_shape: tuple[int, int, int],
    ) -> np.ndarray:
        img = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return transform.resize(image=img, output_shape=img_shape)

    def data_augmentation(
        self,
        augment_size: int = 5_000,
    ) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=5,
            zoom_range=0.08,
            width_shift_range=0.08,
            height_shift_range=0.08,
        )
        # Fit the data generator
        image_generator.fit(
            self.x_train,
            augment=True,
        )
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,
            shuffle=False,
        ).next()[0]
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]


if __name__ == "__main__":
    data_dir = os.path.join("C:/Users/Jan/Documents/DogsAndCats")
    d = DOGSCATS(data_dir=data_dir)
