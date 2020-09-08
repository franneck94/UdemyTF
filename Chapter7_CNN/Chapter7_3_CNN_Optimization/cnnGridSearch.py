import os

from sklearn.model_selection import GridSearchCV, ParameterGrid
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPool2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from mnistDataValidation import MNIST


mnist = MNIST()
mnist.data_augmentation(augment_size=5000)
mnist.data_preprocessing(preprocess_mode="MinMax")
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

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


epochs = 3
batch_size = 128
optimizer_candidates = [Adam, RMSprop]
lr_candidates = [1e-3, 5e-3, 1e-4]

param_grid = {
    "optimizer": optimizer_candidates,
    "learning_rate": lr_candidates,
}

grid = ParameterGrid(param_grid)
for comb in grid:
    print(comb)

keras_clf = KerasClassifier(build_fn=model_fn, epochs=epochs, batch_size=batch_size, verbose=0)

grid_cv = GridSearchCV(estimator=keras_clf, param_grid=param_grid, n_jobs=1, verbose=0, cv=3)

grid_result = grid_cv.fit(x_train, y_train)

# Summary
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, std, param in zip(means, stds, params):
    print("Acc: %f (+/- %f) with: %r" % (mean, std, param))
