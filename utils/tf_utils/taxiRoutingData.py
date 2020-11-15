import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class TAXIROUTING:
    def __init__(self, excel_file_path: str) -> None:
        # Load the excel file
        self.column_names = [
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
        self.df = pd.read_excel(open(excel_file_path, "rb"))
        self.df = pd.DataFrame(data=self.df, columns=self.column_names)
        self.feature_names = [
            "Uhrzeit",
            "Lat Start",
            "Lon Start",
            "Lat Ziel",
            "Lon Ziel",
            "OSRM Dauer",
            "OSRM Distanz",
        ]
        self.x = self.df.loc[:, self.feature_names]
        self.y = self.df["y"]
        self.x = self.x.to_numpy()
        self.y = self.y.to_numpy()
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
        # scaler = MinMaxScaler()
        scaler.fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)


if __name__ == "__main__":
    excel_file_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/data/taxiDataset.xlsx")
    taxi_data = TAXIROUTING(excel_file_path=excel_file_path)

    df = pd.DataFrame(data=taxi_data.x, columns=taxi_data.feature_names)
    df["y"] = taxi_data.y
    print(df.head())
    print(df.describe())
    print(df.info())
    df.hist(bins=30, figsize=(20, 15))
    plt.show()
