import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401


class CALIHOUSING:
    def __init__(self):
        self.dataset = fetch_california_housing()
        self.feature_names = self.dataset.feature_names
        self.DESCR = self.dataset.DESCR
        self.x = self.dataset.data
        self.y = self.dataset.target
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
        # scaler = StandardScaler()
        # scaler = MinMaxScaler()
        # scaler.fit(self.x_train)
        # self.x_train = scaler.transform(self.x_train)
        # self.x_test = scaler.transform(self.x_test)


if __name__ == "__main__":
    cali_data = CALIHOUSING()
    print(cali_data.x_train.shape, cali_data.y_train.shape)
    print(cali_data.x_test.shape, cali_data.y_test.shape)

    df = pd.DataFrame(data=cali_data.x, columns=cali_data.feature_names)
    df["y"] = cali_data.y

    # print(cali_data.DESCR)
    # print(df.head())
    # print(df.describe())
    # print(df.info())

    # df.hist(bins=30, figsize=(20,15))
    # plt.show()

    df.plot(
        kind="scatter",
        x="Longitude",
        y="Latitude",
        alpha=0.4,
        figsize=(10, 7),
        c="y",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
    )
    plt.show()
