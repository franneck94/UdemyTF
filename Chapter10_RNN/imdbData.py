import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence

class IMDBData:
    def __init__(self, num_words, skip_top, maxlen):
        # Dataset variables
        self.num_classes = 2
        self.num_words = num_words
        self.skip_top = skip_top
        self.maxlen = maxlen
        # Word index: Word -> Index
        self.word_index = imdb.get_word_index()
        self.word_index = {key: (val+3) for key,val in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        # Index -> Word
        self.index_to_word = {val: key for key, val in self.word_index.items()}
        # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(
            num_words=self.num_words,
            skip_top=self.skip_top)
        print(self.x_train[0])
        print(self.y_train[0])

if __name__ == "__main__":
    num_words = 10000
    skip_top = 20
    maxlen = 80
    imdb_data = IMDBData(num_words, skip_top, maxlen)