import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import SimpleRNN
from keras.layers import TextVectorization
from keras.models import Model
from keras.optimizers import Adam
from tensorcross.model_selection import GridSearch

from tf_utils.imdbDataAdvanced import IMDB


np.random.seed(0)  # noqa: NPY002
tf.random.set_seed(0)


def save_embedding(
    vectorizer: TextVectorization,
    embedding_dim: int,
) -> int:
    """
    Glove:
    !wget http://nlp.stanford.edu/data/glove.6B.zip
    !unzip -q glove.6B.zip
    """
    vocab = vectorizer.get_vocabulary()
    word_index = dict(zip(vocab, range(len(vocab))))
    num_tokens = len(vocab)
    embeddings_index = {}
    path_to_glove_file = os.path.join(
        os.path.expanduser("~"),
        f".keras/datasets/glove.6B.{embedding_dim}d.txt",
    )
    with open(path_to_glove_file) as f:  # noqa: PLW1514
        for line in f:
            word, coefs_ = line.split(maxsplit=1)
            coefs = np.fromstring(coefs_, "f", sep=" ")
            embeddings_index[word] = coefs

    print(f"Found {len(embeddings_index)} word vectors.")
    num_tokens = len(vocab)
    hits = 0
    misses = 0
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if (
            embedding_vector is not None
            and len(embedding_vector) == embedding_dim
        ):
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f"Converted {hits} words ({misses} misses)")
    np.save(
        os.path.join(os.path.expanduser("~"), ".keras/datasets/embedding.npy"),
        embedding_matrix,
    )
    return embedding_dim


def load_embedding() -> np.ndarray:
    return np.load(
        os.path.join(os.path.expanduser("~"), ".keras/datasets/embedding.npy")
    )


def create_rnn_model(
    input_shape: tuple[int, int],
    num_classes: int,
    sequence_length: int,
    vocab_size: int,
    embedding_dim: int,
    rec_units1: int,
    rec_units2: int,
    dense_units: int,
    dropout_rate: float,
    embedding_matrix: np.ndarray,
) -> Model:
    input_text = Input(shape=input_shape)
    x = Reshape(target_shape=(input_shape[0],))(input_text)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=True,
    )(x)
    x = Bidirectional(
        SimpleRNN(
            units=rec_units1,
            return_sequences=True,
        )
    )(x)
    x = Bidirectional(
        SimpleRNN(
            units=rec_units2,
            return_sequences=False,
        )
    )(x)
    x = Dense(units=dense_units)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    x = Dropout(rate=dropout_rate)(x)
    out = Activation("softmax")(x)
    model = Model(
        inputs=[input_text],
        outputs=[out],
    )
    optimizer = Adam(learning_rate=1e-3)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    # model.summary()
    return model


def create_gru_model(
    input_shape: tuple[int, int],
    num_classes: int,
    sequence_length: int,
    vocab_size: int,
    embedding_dim: int,
    rec_units1: int,
    rec_units2: int,
    dense_units: int,
    dropout_rate: float,
    embedding_matrix: np.ndarray,
) -> Model:
    input_text = Input(shape=input_shape)
    x = Reshape(target_shape=(input_shape[0],))(input_text)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=True,
    )(x)
    x = Bidirectional(
        GRU(
            units=rec_units1,
            return_sequences=True,
        )
    )(x)
    x = Bidirectional(
        GRU(
            units=rec_units2,
            return_sequences=False,
        )
    )(x)
    x = Dense(units=dense_units)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    x = Dropout(rate=dropout_rate)(x)
    out = Activation("softmax")(x)
    model = Model(
        inputs=[input_text],
        outputs=[out],
    )
    optimizer = Adam(learning_rate=1e-3)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    # model.summary()
    return model


def create_lstm_model(
    input_shape: tuple[int, int],
    num_classes: int,
    sequence_length: int,
    vocab_size: int,
    embedding_dim: int,
    rec_units1: int,
    rec_units2: int,
    dense_units: int,
    dropout_rate: float,
    embedding_matrix: np.ndarray,
) -> Model:
    input_text = Input(shape=input_shape)
    x = Reshape(target_shape=(input_shape[0],))(input_text)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=True,
    )(x)
    x = Bidirectional(
        LSTM(
            units=rec_units1,
            return_sequences=True,
        )
    )(x)
    x = Bidirectional(
        LSTM(
            units=rec_units2,
            return_sequences=False,
        )
    )(x)
    x = Dense(units=dense_units)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    x = Dropout(rate=dropout_rate)(x)
    out = Activation("softmax")(x)
    model = Model(inputs=[input_text], outputs=[out])
    optimizer = Adam(learning_rate=1e-3)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    # model.summary()
    return model


def main() -> None:
    vocab_size = 20000
    sequence_length = 80
    embedding_dim = 50
    imdb_data = IMDB(vocab_size, sequence_length)

    # save_embedding(imdb_data.vectorize_layer, embedding_dim)
    embedding_matrix = load_embedding()

    num_classes = imdb_data.num_classes
    input_shape = (sequence_length, 1)

    batch_size = 128
    epochs = 10

    model_fns = {
        # "RNN": create_rnn_model,
        "GRU": create_gru_model,
        # "LSTM": create_lstm_model,
    }

    for model_name, model_fn in model_fns.items():
        param_grid = {
            "rec_units1": [80],
            "rec_units2": [60],
            "dense_units": [40],
            "dropout_rate": [0.2],
        }
        kwargs = {
            "input_shape": input_shape,
            "num_classes": num_classes,
            "sequence_length": sequence_length,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "embedding_matrix": embedding_matrix,
        }

        search = GridSearch(
            model_fn=model_fn,
            param_grid=param_grid,
            **kwargs,
        )

        batch_size = 512
        epochs = 100
        es_callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            restore_best_weights=True,
        )

        fit_kwargs = {
            "verbose": 1,
            "batch_size": batch_size,
            "epochs": epochs,
            "callbacks": [es_callback],
        }

        search.fit(
            train_dataset=imdb_data.train_dataset,
            val_dataset=imdb_data.val_dataset,
            **fit_kwargs,
        )

        print(f"Model: {model_name}")
        search.summary()


if __name__ == "__main__":
    main()
