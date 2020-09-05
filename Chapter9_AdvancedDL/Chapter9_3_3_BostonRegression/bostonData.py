import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import boston_housing

class BOSTON:
    def __init__(self):
        # Load the data set
        (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        self.x_train_ = None
        self.x_val = None
        self.y_train_ = None
        self.y_val = None
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
        self.num_targets = 1 # Constant for the data set
        self.num_features = self.x_train.shape[1]
        # Addtional class attributes
        self.scaler = None

    def get_train_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size=0.33):
        self.x_train_, self.x_val, self.y_train_, self.y_val =\
            train_test_split(self.x_train, self.y_train, test_size=validation_size)
        self.val_size = self.x_val.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val, self.y_train_, self.y_val

    def data_preprocessing(self, preprocess_mode="standard", preprocess_params=None):
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
        # Fitting and transforming
        self.scaler.fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
