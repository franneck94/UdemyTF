from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from imdbData2 import *

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


def create_model():
    input_text = Input(shape=x_train.shape[1:])
    x = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen)(input_text)
    x = LSTM(units=100)(x)
    x = Dense(units=num_classes)(x)
    output_pred = Activation("softmax")(x)

    optimizer = Adam(
        lr=1e-3)
    model = Model(
        inputs=input_text,
        outputs=output_pred)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])
    model.summary()
    return model


model = create_model()
model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test))

# Test the DNN
score = model.evaluate(
    x_test,
    y_test,
    verbose=0,
    batch_size=batch_size)
print("Test performance: ", score)
