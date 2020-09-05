import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical


class IMDBData:
    def __init__(self, num_words, skip_top, maxlen):
        # Dataset variables
        self.num_classes = 2
        self.num_words = num_words
        self.skip_top = skip_top
        self.maxlen = maxlen
        # Word index: Word -> Index
        self.word_index = imdb.get_word_index()
        self.word_index = {key: (val + 3) for key, val in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        # Index -> Word
        self.index_to_word = {val: key for key, val in self.word_index.items()}
        # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(
            num_words=self.num_words,
            skip_top=self.skip_top)
        # Save texts
        self.x_train_text = np.array(
            [[self.index_to_word[index] for index in review] for review in self.x_train])
        self.x_test_text = np.array(
            [[self.index_to_word[index] for index in review] for review in self.x_test])
        # Pad sequences
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array for class labels
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def get_review_text(self, review):
        reveiw_text = [self.index_to_word[index] for index in review]
        return reveiw_text


if __name__ == "__main__":
    num_words = 10000
    skip_top = 20
    maxlen = 80
    imdb_data = IMDBData(num_words, skip_top, maxlen)
    print(imdb_data.x_train[1337])
    print(imdb_data.y_train[1337])
