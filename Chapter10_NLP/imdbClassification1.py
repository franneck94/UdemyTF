import numpy as np
import tensorflow as tf
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import SimpleRNN
from keras.models import Model
from keras.optimizers import Adam

from tf_utils.imdbDataAdvanced import IMDB


np.random.seed(0)
tf.random.set_seed(0)


def create_rnn_model(
    input_shape: tuple[int, int],
    num_classes: int,
) -> Model:
    input_text = Input(shape=input_shape)
    x = SimpleRNN(units=80, return_sequences=False)(input_text)
    x = Dense(units=80)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    out = Activation("softmax")(x)
    model = Model(
        inputs=[input_text],
        outputs=[out],
    )
    opt = Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )
    model.summary()
    return model


def create_lstm_model(
    input_shape: tuple[int, int],
    num_classes: int,
) -> Model:
    input_text = Input(shape=input_shape)
    x = LSTM(units=80, return_sequences=False)(input_text)
    x = Dense(units=80)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    out = Activation("softmax")(x)
    model = Model(
        inputs=[input_text],
        outputs=[out],
    )
    opt = Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )
    model.summary()
    return model


def create_gru_model(
    input_shape: tuple[int, int],
    num_classes: int,
) -> Model:
    input_text = Input(shape=input_shape)
    x = GRU(units=80, return_sequences=False)(input_text)
    x = Dense(units=80)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    out = Activation("softmax")(x)
    model = Model(
        inputs=[input_text],
        outputs=[out],
    )
    opt = Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )
    model.summary()
    return model


def main() -> None:
    vocab_size = 20_000
    sequence_length = 80
    imdb_data = IMDB(vocab_size, sequence_length)
    train_dataset = imdb_data.get_train_set()
    val_dataset = imdb_data.get_val_set()
    test_dataset = imdb_data.get_test_set()
    input_shape = (sequence_length, 1)
    num_classes = imdb_data.num_classes

    batch_size = 512
    epochs = 10

    model_fns = {
        "RNN": create_rnn_model,
        "LSTM": create_lstm_model,
        "GRU": create_gru_model,
    }

    for name, model_fn in model_fns.items():
        print(f"Model: {name}")
        model = model_fn(input_shape, num_classes)
        model.fit(
            x=train_dataset,
            verbose=1,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_dataset,
        )
        score = model.evaluate(
            x=test_dataset,
            verbose=0,
            batch_size=batch_size,
        )
        print(f"Test performance: {score}")


if __name__ == "__main__":
    main()
