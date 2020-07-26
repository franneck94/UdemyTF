import os

import random
random.seed(0)

import numpy as np
np.random.seed(0)

from sklearn.model_selection import RandomizedSearchCV, ParameterSampler

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from plotting import *
from mnistDataValidation import *

mnist = MNIST()
mnist.data_augmentation(augment_size=5000)
mnist.data_preprocessing(preprocess_mode="MinMax")
x_train, y_train = mnist.get_train_set()
x_test, y_test = mnist.get_test_set()
num_classes = mnist.num_classes

# Save Path
dir_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTensorflowKurs/models/")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
mnist_model_path = os.path.join(dir_path, "mnist_model.h5")
# Log Dir
log_dir = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTensorflowKurs/logs/")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
model_log_dir = os.path.join(log_dir, "modelMinMaxFinal")

def model_fn(optimizer, learning_rate):
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
    x = Dense(units=128)(x)
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

epochs = 3
batch_size = 128
optimizer_candidates = [Adam, RMSprop]
lr_candidates = [random.uniform(1e-4, 1e-3) for _ in range(100)]

param_distributions = {
    "optimizer": optimizer_candidates,
    "learning_rate": lr_candidates,
}

keras_clf = KerasClassifier(
    build_fn=model_fn,
    epochs=epochs,
    batch_size=batch_size,
    verbose=0)

rand_cv = RandomizedSearchCV(
    estimator=keras_clf,
    param_distributions=param_distributions,
    n_iter=4,
    n_jobs=1,
    verbose=0,
    cv=3)

rand_result = rand_cv.fit(x_train, y_train)

# Summary
print("Best: %f using %s" % (rand_result.best_score_, rand_result.best_params_))

means = rand_result.cv_results_["mean_test_score"]
stds = rand_result.cv_results_["std_test_score"]
params = rand_result.cv_results_["params"]

for mean, std, param in zip(means, stds, params):
    print("Acc: %f (+/- %f) with: %r" % (mean, std, param))
