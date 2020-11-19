from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)


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
        "Lat Start",
        "Lon Start",
        "Lat Ziel",
        "Lon Ziel",
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
    def __init__(self, excel_file_path: str) -> None:
        # Load the dataset
        dataset = load_dataset(excel_file_path)
        self.x = dataset["data"]
        self.y = dataset["target"]
        self.feature_names = dataset["feature_names"]
        # User-definen constants
        self.num_targets = 1
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)
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

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_val, self.y_val
