import os
from typing import Tuple

import cv2
import numpy as np
from skimage import transform
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


FILE_DIR = os.path.abspath("C:/Users/Jan/Documents/DogsAndCats")
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_DEPTH = 3


def extract_cats_vs_dogs() -> None:
    cats_dir = os.path.join(FILE_DIR, "cat")
    dogs_dir = os.path.join(FILE_DIR, "dog")

    print("Deleting no .jpg images!")
    for f in os.listdir(cats_dir):
        if f.split(".")[-1] != "jpg":
            print("Removing file: ", f)
            os.remove(os.path.join(cats_dir, f))
    for f in os.listdir(dogs_dir):
        if f.split(".")[-1] != "jpg":
            print("Removing file: ", f)
            os.remove(os.path.join(dogs_dir, f))

    num_cats = len(os.listdir(cats_dir))
    num_dogs = len(os.listdir(dogs_dir))
    num_images = num_cats + num_dogs

    x = np.zeros(shape=(num_images, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), dtype=np.float32)
    y = np.zeros(shape=(num_images), dtype=np.int8)

    cnt = 0
    print("Start reading cat images!")
    for f in os.listdir(cats_dir):
        img_file = os.path.join(cats_dir, f)
        try:
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x[cnt] = transform.resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
            y[cnt] = 0
            cnt += 1
        except:
            print("Cat image %s cannot be read!" % f)
            os.remove(img_file)

    print("Start reading dog images!")
    for f in os.listdir(dogs_dir):
        img_file = os.path.join(dogs_dir, f)
        try:
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x[cnt] = transform.resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
            y[cnt] = 1
            cnt += 1
        except:
            print("Dog image %s cannot be read!" % f)
            os.remove(img_file)

    # Dropping not readable image idxs
    x = x[:cnt]
    y = y[:cnt]

    np.save(os.path.join(FILE_DIR, "x.npy"), x)
    np.save(os.path.join(FILE_DIR, "y.npy"), y)


def load_cats_vs_dogs(test_size: float = 0.33, extracting_images: bool = False) -> tuple:
    file_x = os.path.join(FILE_DIR, "x.npy")
    file_y = os.path.join(FILE_DIR, "y.npy")

    if not os.path.isfile(file_x) or not os.path.isfile(file_y) or extracting_images:
        extract_cats_vs_dogs()

    x = np.load(file_x)
    y = np.load(file_y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)


class DOGSCATS:
    def __init__(self, test_size: float = 0.33, extracting_images: bool = False) -> None:
        # Load the data set
        # Class 0: Cat, Class 1: Dog
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cats_vs_dogs(
            test_size=test_size, extracting_images=extracting_images
        )
        self.x_train_: np.ndarray = None
        self.x_val: np.ndarray = None
        self.y_train_: np.ndarray = None
        self.y_val: np.ndarray = None
        # Convert to float32
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # Save important data attributes as variables
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.train_splitted_size = 0
        self.val_size = 0
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.num_classes = 2  # Constant for the data set
        self.num_features = self.width * self.height * self.depth
        # Reshape the y data to one hot encoding
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        # Addtional class attributes
        self.scaler: TransformerMixin = None

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size: float = 0.33) -> tuple:
        self.x_train_, self.x_val, self.y_train_, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=validation_size
        )
        self.val_size = self.x_val.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val, self.y_train_, self.y_val

    def data_augmentation(self, augment_size: int = 5_000) -> None:
        # Create an instance of the image data generator class
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            fill_mode='constant',
            cval=0.0,
        )
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(
            x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False
        ).next()[0]
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]

    def data_preprocessing(self, preprocess_mode: str = "standard", preprocess_params: dict = None) -> None:
        # Preprocess the data
        if preprocess_mode == "standard":
            if preprocess_params:
                self.scaler = StandardScaler(**preprocess_params)
            else:
                self.scaler = StandardScaler(**preprocess_params)
        else:
            if preprocess_params:
                self.scaler = MinMaxScaler(**preprocess_params)
            else:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Temporary flatteining of the x data
        self.x_train = self.x_train.reshape(self.train_size, self.num_features)
        self.x_test = self.x_test.reshape(self.test_size, self.num_features)
        # Fitting and transforming
        self.scaler.fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        # Reshaping the xdata back to the input shape
        self.x_train = self.x_train.reshape((self.train_size, self.width, self.height, self.depth))
        self.x_test = self.x_test.reshape((self.test_size, self.width, self.height, self.depth))
