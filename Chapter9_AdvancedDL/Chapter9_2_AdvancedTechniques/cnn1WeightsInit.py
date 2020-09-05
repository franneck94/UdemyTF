import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
random.seed(0)

import numpy as np
np.random.seed(0)

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from plotting import *
from dogsCatsData import *

data = DOGSCATS()
data.data_augmentation(augment_size=5000)
data.data_preprocessing(preprocess_mode="MinMax")
x_train_splitted, x_val, y_train_splitted, y_val = data.get_splitted_train_validation_set()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()
num_classes = data.num_classes

# Save Path
dir_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTensorflowKurs/models/")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
data_model_path = os.path.join(dir_path, "data_model.h5")
# Log Dir
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTensorflowKurs/logs/")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# Define the DNN
def model_fn(optimizer, learning_rate, filter_block1, kernel_size_block1, filter_block2, 
             kernel_size_block2, filter_block3, kernel_size_block3, dense_layer_size,
             kernel_initializer, bias_initializer):
    # Input
    input_img = Input(shape=x_train.shape[1:])
    # Conv Block 1
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same',
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same',
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Conv Block 2
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same',
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same',
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Conv Block 3
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same',
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same',
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Dense Part
    x = Flatten()(x)
    x = Dense(units=dense_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"])
    return model

# Global params
epochs = 20
batch_size = 256

optimizer = Adam
learning_rate = 0.001
filter_block1 = 32
kernel_size_block1 = 3
filter_block2 = 64
kernel_size_block2 = 3
filter_block3 = 128
kernel_size_block3 = 3
dense_layer_size = 512
# GlorotUniform, GlorotNormal, RandomNormal, RandomUniform, VarianceScaling
kernel_initializer = 'GlorotUniform'
bias_initializer = 'zeros'

rand_model = model_fn(optimizer, learning_rate, filter_block1, kernel_size_block1, filter_block2, 
                      kernel_size_block2, filter_block3, kernel_size_block3, dense_layer_size,
                      kernel_initializer, bias_initializer)
model_log_dir = os.path.join(log_dir, "modelCatsDogsVarianceScaling")
tb_callback = TensorBoard(
    log_dir=model_log_dir)
rand_model.fit(
    x=x_train_splitted, 
    y=y_train_splitted, 
    verbose=1, 
    batch_size=batch_size, 
    epochs=epochs, 
    callbacks=[tb_callback],
    validation_data=(x_val, y_val))
score = rand_model.evaluate(
    x_test, 
    y_test, 
    verbose=0, 
    batch_size=batch_size)
print("Test performance: ", score)
