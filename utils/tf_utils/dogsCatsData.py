import os
from typing import Tuple

import cv2
import numpy as np
from skimage import transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


DATA_DIR = os.path.abspath("C:/Users/Jan/Documents/DogsAndCats")
X_FILE_PATH = os.path.join(DATA_DIR, "x.npy")
Y_FILE_PATH = os.path.join(DATA_DIR, "y.npy")
IMG_SIZE = 64
IMG_DEPTH = 3


def extract_cats_vs_dogs() -> None:
    cats_dir = os.path.join(DATA_DIR, "cat")
    dogs_dir = os.path.join(DATA_DIR, "dog")

    dirs = [cats_dir, dogs_dir]
    class_names = ["cats", "dogs"]

    for d in dirs:
        for f in os.listdir(d):
            if f.split(".")[-1] != "jpg":
                print(f"Removing file: {f}")
                os.remove(os.path.join(d, f))

    num_cats = len(os.listdir(cats_dir))
    num_dogs = len(os.listdir(dogs_dir))
    num_images = num_cats + num_dogs

    x = np.zeros(
        shape=(num_images, IMG_SIZE, IMG_SIZE, IMG_DEPTH),
        dtype=np.float32
    )
    y = np.zeros(
        shape=(num_images),
        dtype=np.int8
    )

    cnt = 0
    for d, class_name in zip(dirs, class_names):
        for f in os.listdir(d):
            img_file = os.path.join(d, f)
            try:
                img = cv2.imread(img_file, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x[cnt] = transform.resize(
                    image=img,
                    output_shape=(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
                )
                if class_name == "cats":
                    y[cnt] = 0
                elif class_name == "dogs":
                    y[cnt] = 1
                else:
                    raise ValueError
                cnt += 1
            except:  # noqa: E722
                print(f"Image {f} cannot be read!")
                os.remove(img_file)

    # Dropping not readable image idxs
    x = x[:cnt]
    y = y[:cnt]

    np.save(X_FILE_PATH, x)
    np.save(Y_FILE_PATH, y)


class MNIST:
    def __init__(self, with_normalization: bool = True, test_size: float = 0.2, validation_size: float = 0.33) -> None:
        # Helper variables
        self.num_classes = 2
        # Load the data set
        x = np.load(X_FILE_PATH)
        y = np.load(Y_FILE_PATH)
        y = to_categorical(y, num_classes=self.num_classes)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        self.num_classes = 10
        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size: float = 0.33) -> tuple:
        self.x_train_, self.x_val_, self.y_train_, self.y_val_ = train_test_split(
            self.x_train,
            self.y_train,
            test_size=validation_size
        )
        self.val_size = self.x_val_.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val_, self.y_train_, self.y_val_

    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=5,
            zoom_range=0.08,
            width_shift_range=0.08,
            height_shift_range=0.08
        )
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,
            shuffle=False
        ).next()[0]
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
