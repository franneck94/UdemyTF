import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

# Dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
train_size = x_train.shape[0]
test_size = x_test.shape[0]

regr = LinearRegression()
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
y_pred = regr.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Score: ", score)
print("MSE: ", mse)