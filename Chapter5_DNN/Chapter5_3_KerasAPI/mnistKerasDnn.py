import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant, TruncatedNormal
from tensorflow.keras.layers import Activation, Dense
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
num_features = 784
num_classes = 10

# Compute the categorical classes
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape the input data
x_train = x_train.reshape(train_size, num_features)
x_test = x_test.reshape(test_size, num_features)

# Model params
init_w = TruncatedNormal(mean=0.0, stddev=0.01)
init_b = Constant(value=0.0)
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 10
batch_size = 256  # [32, 1024]

# Define the DNN
model = Sequential()

model.add(
    Dense(
        units=500,
        kernel_initializer=init_w,
        bias_initializer=init_b,
        input_shape=(num_features,),
    )
)
model.add(Activation("relu"))

model.add(Dense(units=300, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation("relu"))

model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation("relu"))

model.add(Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b))
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
