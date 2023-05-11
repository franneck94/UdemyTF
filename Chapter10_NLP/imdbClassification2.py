from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import SimpleRNN
from keras.models import Model
from keras.optimizers import Adam

from tensorcross.model_selection import GridSearch
from tf_utils.imdbDataAdvanced import IMDB


np.random.seed(0)
tf.random.set_seed(0)


def create_rnn_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    sequence_length: int,
    vocab_size: int,
    embedding_dim: int,
    rec_units: int,
    dense_units: int,
    dropout_rate: float,
) -> Model:
    input_text = Input(shape=input_shape)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
    )(input_text)
    x = SimpleRNN(units=rec_units)(x)
    x = Dense(units=dense_units)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    x = Dropout(rate=dropout_rate)(x)
    out = Activation("softmax")(x)
    model = Model(inputs=[input_text], outputs=[out])
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.summary()
    return model


def create_gru_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    sequence_length: int,
    vocab_size: int,
    embedding_dim: int,
    rec_units: int,
    dense_units: int,
    dropout_rate: float,
) -> Model:
    input_text = Input(shape=input_shape)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
    )(input_text)
    x = GRU(units=rec_units)(x)
    x = Dense(units=dense_units)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    x = Dropout(rate=dropout_rate)(x)
    out = Activation("softmax")(x)
    model = Model(inputs=[input_text], outputs=[out])
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.summary()
    return model


def create_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    sequence_length: int,
    vocab_size: int,
    embedding_dim: int,
    rec_units: int,
    dense_units: int,
    dropout_rate: float,
) -> Model:
    input_text = Input(shape=input_shape)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
    )(input_text)
    x = LSTM(units=rec_units)(x)
    x = Dense(units=dense_units)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    x = Dropout(rate=dropout_rate)(x)
    out = Activation("softmax")(x)
    model = Model(inputs=[input_text], outputs=[out])
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.summary()
    return model


def main() -> None:
    vocab_size = 20000
    sequence_length = 80
    embedding_dim = 50
    imdb_data = IMDB(vocab_size, sequence_length)

    num_classes = imdb_data.num_classes
    input_shape = (sequence_length,)

    model_fns = {
        # "RNN": create_rnn_model,
        "GRU": create_gru_model,
        "LSTM": create_lstm_model,
    }

    for model_name, model_fn in model_fns.items():
        param_grid = {
            "rec_units": [60],
            "dense_units": [60],
            "dropout_rate": [0.0],
        }
        kwargs = dict(
            input_shape=input_shape,
            num_classes=num_classes,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        search = GridSearch(model_fn=model_fn, param_grid=param_grid, **kwargs)

        batch_size = 512
        epochs = 100
        es_callback = EarlyStopping(
            monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
        )

        fit_kwargs = dict(
            verbose=0,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[es_callback],
        )

        search.fit(
            train_dataset=imdb_data.train_dataset,
            val_dataset=imdb_data.val_dataset,
            **fit_kwargs,
        )

        print(f"Model: {model_name}")
        search.summary()


if __name__ == "__main__":
    main()
