import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from caliHousingData import *

cali_data = CALIHOUSING()
x_train, y_train = cali_data.x_train, cali_data.y_train
x_test, y_test = cali_data.x_test, cali_data.y_test
num_features = cali_data.num_features
num_targets = cali_data.num_targets

regr = LinearRegression()
regr.fit(x_train, y_train) # Training
print(regr.score(x_test, y_test)) # Testing