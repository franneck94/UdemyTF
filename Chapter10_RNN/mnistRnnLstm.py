import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to np.float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Dataset variables
train_size = x_train.shape[0]
test_size = x_test.shape[0]
num_timesteps = 28
num_features = 28
num_classes = 10

# Compute the categorical classes
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape the input data
x_train = x_train.reshape(train_size, num_timesteps, num_features)
x_test = x_test.reshape(test_size, num_timesteps, num_features)

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 10
batch_size = 256
units = 50
return_sequences = False

# Define the DNN
model = Sequential()
model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Dense(units=num_classes))
model.add(Activation("softmax"))
model.summary()

# Compile and train (fit) the model, afterwards evaluate the model
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.fit(
    x=x_train,
    y=y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Score: ", score)
