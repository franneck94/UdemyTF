from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_taxi_dataset(excel_file_path) -> Dict[str, Any]:
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
    x = df.loc[:, feature_names].to_numpy()
    y = df["y"].to_numpy()
    return {
        "feature_names": column_names,
        "data": x,
        "target": y
    }


class TAXIROUTING:
    def __init__(self, excel_file_path: str) -> None:
        # Load the dataset
        dataset = load_taxi_dataset(excel_file_path)
        self.x = dataset["data"]
        self.y = dataset["target"]
        self.feature_names = dataset["feature_names"]
        # Prepare x data
        self.x[:, 0] = [float(val[:2]) * 60 + float(val[3:5]) for val in self.x[:, 0]]
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)
        # Split the dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.train_size, self.test_size = self.x_train.shape[0], self.x_test.shape[0]
        # Change dtype from int to float
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # Reshape the y-data
        self.y_train = np.reshape(self.y_train, (self.train_size, 1))
        self.y_test = np.reshape(self.y_test, (self.test_size, 1))
        # Dataset variables
        self.num_features = self.x_train.shape[1]
        self.num_targets = self.y_train.shape[1]
        # Data rescaling
        scaler = StandardScaler()
        scaler.fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
