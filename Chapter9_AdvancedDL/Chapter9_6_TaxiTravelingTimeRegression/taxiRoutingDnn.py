import os

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from taxiRoutingData import TAXIROUTING


excel_file_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/data/taxiDataset.xlsx")
taxi_data = TAXIROUTING(excel_file_path=excel_file_path)
x_train, y_train = taxi_data.x_train, taxi_data.y_train
x_test, y_test = taxi_data.x_test, taxi_data.y_test
num_features = taxi_data.num_features
num_targets = taxi_data.num_targets


def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 200
batch_size = 256

# Define the DNN
model = Sequential()
model.add(Dense(units=16, input_shape=(num_features,)))
model.add(Activation("relu"))
model.add(Dense(units=16))
model.add(Activation("relu"))
model.add(Dense(units=num_targets))
model.summary()

# Compile and train (fit) the model, afterwards evaluate the model
model.compile(loss="mse", optimizer=optimizer, metrics=[r_squared])
model.fit(
    x=x_train,
    y=y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
)
score = model.evaluate(x_test, y_test, verbose=0)
print("Score: ", score)
