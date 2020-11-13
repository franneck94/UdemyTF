from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from imdbData2 import IMDBData


def create_model() -> Model:
    input_text = Input(shape=x_train.shape[1:])
    x = Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        input_length=maxlen
    )(input_text)
    x = LSTM(units=100)(x)
    x = Dense(units=num_classes)(x)
    output_pred = Activation("softmax")(x)

    optimizer = Adam(learning_rate=1e-3)
    model = Model(
        inputs=input_text,
        outputs=output_pred
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    model.summary()
    return model


if __name__ == "__main__":
    # Load MNIST dataset
    num_words = 10000
    skip_top = 100
    maxlen = 100
    embedding_dim = 10
    imdb_data = IMDBData(num_words, skip_top, maxlen)
    x_train, y_train = imdb_data.x_train, imdb_data.y_train
    x_test, y_test = imdb_data.x_test, imdb_data.y_test

    num_classes = imdb_data.num_classes
    batch_size = 256
    epochs = 10

    model = create_model()

    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
    )

    score = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0,
        batch_size=batch_size
    )
    print(f"Test performance: {score}")
