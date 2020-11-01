import os

import numpy as np
import tensorflow as tf


FILE_PATH = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
MODEL_PATH = os.path.join(PROJECT_PATH, "ressources", "weights", "dnn_mnist.h5")


def create_model():
    # Model params
    num_features = 784
    num_classes = 10
    init_w = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    init_b = tf.keras.initializers.Constant(value=0.0)
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define the DNN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=500, kernel_initializer=init_w,
                                    bias_initializer=init_b, input_shape=(num_features,)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(units=300, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(tf.keras.layers.Activation("softmax"))

    # Compile and train (fit) the model, afterwards evaluate the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


def nn_predict(model, image=None):
    if image is not None and model is not None:
        pred = model.predict(image.reshape(1, 784))[0]
        pred = np.argmax(pred, axis=0)
        return pred
    else:
        return -1


def nn_train():
    # Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    # Reshape the input data
    x_train = x_train.reshape(train_size, num_features)
    x_test = x_test.reshape(test_size, num_features)

    epochs = 10
    batch_size = 256

    model = create_model()

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    model.save_weights(MODEL_PATH)


if __name__ == "__main__":
    nn_train()
