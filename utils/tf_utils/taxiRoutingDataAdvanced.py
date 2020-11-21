from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import Normalization


np.random.seed(0)
tf.random.set_seed(0)


def load_dataset(excel_file_path) -> Dict[str, Any]:
    # Load the excel file
    column_names = [
        "Uhrzeit",
        "Straße Start",
        "Nr Start",
        "Stadt Start",
        "Lat Start",
        "Lon Start",
        "Straße Ziel",
        "Nr Ziel",
        "Stadt Ziel",
        "Lat Ziel",
        "Lon Ziel",
        "OSRM Dauer",
        "OSRM Distanz",
        "y",
    ]
    df = pd.read_excel(open(excel_file_path, "rb"))
    df = pd.DataFrame(data=df, columns=column_names)
    feature_names = [
        "Uhrzeit",
        # "Lat Start",
        # "Lon Start",
        # "Lat Ziel",
        # "Lon Ziel",
        "OSRM Dauer",
        "OSRM Distanz",
    ]
    x = df[feature_names].to_numpy()
    x[:, 0] = [float(val[:2]) * 60 + float(val[3:5]) for val in x[:, 0]]
    y = df["y"].to_numpy()
    return {
        "feature_names": feature_names,
        "data": x,
        "target": y
    }


class TAXIROUTING:
    def __init__(self, excel_file_path: str, test_size: float = 0.2, validation_size: float = 0.33) -> None:
        # Load the dataset
        dataset = load_dataset(excel_file_path)
        self.x = dataset["data"]
        self.y = dataset["target"]
        self.feature_names = dataset["feature_names"]
        # User-definen constants
        self.num_targets = 1
        self.batch_size = 128
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)
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
