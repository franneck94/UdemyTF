import os

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from mnistDataValidation import MNIST


mnist = MNIST()
mnist.data_augmentation(augment_size=5000)
mnist.data_preprocessing(preprocess_mode="MinMax")
(
    x_train_splitted,
    x_val,
    y_train_splitted,
    y_val,
) = mnist.get_splitted_train_validation_set()
x_train, y_train = mnist.get_train_set()
x_test, y_test = mnist.get_test_set()
num_classes = mnist.num_classes

# Save Path
dir_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/models/")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
mnist_model_path = os.path.join(dir_path, "mnist_model.h5")
# Log Dir
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
model_log_dir = os.path.join(log_dir, "modelMinMaxFinal")

# Model 1: MinMax, 3 Conv Blocks, Dense: 128, LR=0.001
# Model 2: MinMax, 3 Conv Blocks, Dense: 64,  LR=0.001
# Model 3: MinMax, 3 Conv Blocks, Dense: 256, LR=0.001
# Model 4: MinMax, 3 Conv Blocks, Dense: 256, LR=0.005
# Model 5: MinMax, 3 Conv Blocks, Dense: 256, LR=0.0005

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 30
batch_size = 128

# Define the DNN
input_img = Input(shape=x_train.shape[1:])

x = Conv2D(filters=32, kernel_size=3, padding='same')(input_img)
x = Activation("relu")(x)
x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
x = Activation("relu")(x)
x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
x = Activation("relu")(x)
x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(units=256)(x)
x = Activation("relu")(x)
x = Dense(units=num_classes)(x)
y_pred = Activation("softmax")(x)

# Build the model
model = Model(inputs=[input_img], outputs=[y_pred])

# Compile and train (fit) the model, afterwards evaluate the model
model.summary()

tb_callback = TensorBoard(log_dir=model_log_dir)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, callbacks=[tb_callback])

score = model.evaluate(x_test, y_test, verbose=0)
print("Score: ", score)
