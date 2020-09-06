import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

random.seed(0)

import numpy as np

np.random.seed(0)

import tensorflow as tf

tf.random.set_seed(0)

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
(
    x_train_splitted,
    x_val,
    y_train_splitted,
    y_val,
) = data.get_splitted_train_validation_set()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()
num_classes = data.num_classes

# Save Path
dir_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/models/")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
data_model_path = os.path.join(dir_path, "dogs_cats_plot.h5")
# Log Dir
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


def model_fn(
    optimizer,
    learning_rate,
    filter_block1,
    kernel_size_block1,
    filter_block2,
    kernel_size_block2,
    filter_block3,
    kernel_size_block3,
    dense_layer_size,
    kernel_initializer,
    bias_initializer,
    activation_str,
    dropout_rate,
    use_bn,
    use_global_pooling,
    use_additional_dense_layer,
):
    # Input
    input_img = Input(shape=x_train.shape[1:])
    # Conv Block 1
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name="heatmap",
    )(input_img)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block1,
        kernel_size=kernel_size_block1,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 2
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block2,
        kernel_size=kernel_size_block2,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(
        filters=filter_block3,
        kernel_size=kernel_size_block3,
        padding='same',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == "LeakyReLU":
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Dense Part
    if use_global_pooling:
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    if use_additional_dense_layer:
        x = Dense(units=dense_layer_size)(x)
        if activation_str == "LeakyReLU":
            x = LeakyReLU()(x)
        else:
            x = Activation(activation_str)(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# Global params
epochs = 10
batch_size = 256

params = {
    "optimizer": Adam,
    "learning_rate": 0.001,
    "filter_block1": 32,
    "kernel_size_block1": 3,
    "filter_block2": 64,
    "kernel_size_block2": 3,
    "filter_block3": 128,
    "kernel_size_block3": 3,
    "dense_layer_size": 1024,
    # GlorotUniform, GlorotNormal, RandomNormal, RandomUniform, VarianceScaling
    "kernel_initializer": 'GlorotUniform',
    "bias_initializer": 'zeros',
    # relu, elu, LeakyReLU
    "activation_str": "relu",
    # 0.05, 0.1, 0.2
    "dropout_rate": 0.00,
    # True, False
    "use_bn": True,
    # True, False
    "use_global_pooling": True,
    # True, False
    "use_additional_dense_layer": True,
}

model = model_fn(**params)

plateau_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.95, patience=2, verbose=1, min_lr=1e-5)

es_callback = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, restore_best_weights=True)

# model.fit(
#     x=x_train,
#     y=y_train,
#     verbose=1,
#     batch_size=batch_size,
#     epochs=epochs,
#     callbacks=[plateau_callback, es_callback],
#     validation_data=(x_test, y_test))
# model.save_weights(filepath=data_model_path)

model.load_weights(filepath=data_model_path)
# score = model.evaluate(
#     x_test,
#     y_test,
#     verbose=0,
#     batch_size=batch_size)
# print("Test performance: ", score)

get_occlusion(img=x_test[1337], label=y_test[1337], box_size=4, step_size=4, model=model)
