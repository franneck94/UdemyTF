import os
import shutil
from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

from tf_utils.cifar10DataAdvanced import CIFAR10


np.random.seed(0)
tf.random.set_seed(0)


LOGS_DIR = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/logs/")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def build_model(
    img_shape: Tuple[int, int, int],
    num_classes: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float,
    filter_block1: int,
    kernel_size_block1: int,
    filter_block2: int,
    kernel_size_block2: int,
    filter_block3: int,
    kernel_size_block3: int,
    dense_layer_size: int,
) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same')(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=dense_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    data = CIFAR10()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    img_shape = data.img_shape
    num_classes = data.num_classes

    epochs = 30
    batch_size = 256
    optimizers = [Adam, RMSprop]
    learning_rates = uniform(1e-3, 1e-4)
    filters_block1 = randint(16, 64)
    kernel_sizes_block1 = randint(3, 7)
    filters_block2 = randint(16, 64)
    kernel_sizes_block2 = randint(3, 7)
    filters_block3 = randint(16, 64)
    kernel_sizes_block3 = randint(3, 7)
    dense_layer_sizes = randint(128, 1024)

    param_dist = {
        "optimizer": optimizers,
        "learning_rate": learning_rates,
        "filter_block1": filters_block1,
        "kernel_size_block1": kernel_sizes_block1,
        "filter_block2": filters_block2,
        "kernel_size_block2": kernel_sizes_block2,
        "filter_block3": filters_block3,
        "kernel_size_block3": kernel_sizes_block3,
        "dense_layer_size": dense_layer_sizes,
    }

    n_models = 32
    results = {"best_score": -np.inf, "best_params": {}, "val_scores": [], "params": []}
    rand_search = iter(ParameterSampler(param_dist, n_iter=n_models))

    print(f"Parameter combinations in total: {n_models}")
    for idx in range(n_models):
        comb = next(rand_search)
        print(f"Running comb {idx}")
        curr_model = build_model(
            img_shape=img_shape,
            num_classes=num_classes,
            **comb
        )

        model_log_dir = os.path.join(LOGS_DIR, "modelRand%d" % idx)
        if os.path.exists(model_log_dir):
            shutil.rmtree(model_log_dir)
            os.mkdir(model_log_dir)

        tb_callback = TensorBoard(
            log_dir=model_log_dir,
            histogram_freq=0,
            profile_batch=0
        )

        curr_model.fit(
            train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_dataset,
            callbacks=[tb_callback],
            verbose=1,
        )

        val_accuracy = curr_model.evaluate(
            val_dataset,
            batch_size=256,
            verbose=0
        )[1]
        results["val_scores"].append(val_accuracy)
        results["params"].append(comb)

    best_run_idx = np.argmax(results["val_scores"])
    results["best_score"] = results["val_scores"][best_run_idx]
    results["best_params"] = results["params"][best_run_idx]

    # Summary
    print(f"Best: {results['best_score']} using {results['best_params']}\n")

    scores = results["val_scores"]
    params = results["params"]

    for score, param in zip(scores, params):
        print(f"Acc: {score} with: {param}")
