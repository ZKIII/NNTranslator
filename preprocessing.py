import os
import re
import csv
import random
import jieba
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Helper functions
def regex_text(text: str):
    """
    Apply the regex pattern to the text.

    :param text: string with the text of the dataset.
    :return: string with the regex text.
    """
    reg_text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s+]', '', text)
    reg_text = re.sub(r'\s+', ' ', reg_text)
    reg_text = "<s> " + reg_text + " <e>"
    return reg_text


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    :param seed: int, the seed value.
    :return: None
    """
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_gpu() -> None:
    """
    Set GPU or CPU to train the model.

    :return: None
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU available")
        except RuntimeError as e:
            print(e)
    else:
        print('No GPU available, using CPU instead.')


# Preprocessor Class
class Preprocessor:
    """
    Preprocessing the datasets.
    """

    def __init__(self, path: str):
        """
        Initializing the preprocessor.

        :param path: string with the path of the dataset.
        """
        self.path = path

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.x_tk = Tokenizer(filters='')
        self.y_tk = Tokenizer(filters='')

        self.x_train_seq = None
        self.y_train_seq = None
        self.x_test_seq = None
        self.y_test_seq = None
        self.x_len = None
        self.y_len = None

        # Load and processing data.
        self.load_data()
        self.tokenize()

    def __str__(self):
        """
        Print the parameters information of the preprocessor.

        :return: None
        """
        # The size of each dataset.
        x_train_size = f"The size of x_train: {len(self.x_train)}\n"
        y_train_size = f"The size of y_train: {len(self.y_train)}\n"
        x_test_size = f"The size of x_test: {len(self.x_test)}\n"
        y_test_size = f"The size of y_test: {len(self.y_test)}\n\n"
        size = x_train_size + y_train_size + x_test_size + y_test_size

        # The shape of padded datasets sequences.
        x_train_shape = f"The shape of x_train_seq: {self.x_train_seq.shape}\n"
        y_train_shape = f"The shape of y_train_seq: {self.y_train_seq.shape}\n"
        x_test_shape = f"The shape of x_test_seq: {self.x_test_seq.shape}\n"
        y_test_shape = f"The shape of y_test_seq: {self.y_test_seq.shape}\n\n"
        shape = x_train_shape + y_train_shape + x_test_shape + y_test_shape

        # English and Chinese vocab size.
        x_vocab, y_vocab = self.get_vocab_size()
        vocab = f"EN vocab size: {x_vocab}\nZH vocab size: {y_vocab}\n"

        return size + shape + vocab

    def load_data(self, split_ratio=0.8):
        """
        Load the dataset and split it into train and test set.

        :param split_ratio: float value between 0 and 1.
        :return: None
        """
        x = []
        y = []

        # Load dataset from cmn.txt
        jieba.setLogLevel(jieba.logging.INFO)
        with open(f"{self.path}/cmn.txt", 'r', encoding="utf-8") as cmn_file:
            for line in cmn_file.readlines():
                # English dataset
                input_line, target_line = line.lower().strip().split('\t')
                input_line = regex_text(input_line)
                x.append(input_line)

                # Chinese dataset
                target_line = ' '.join(jieba.cut(target_line))
                target_line = regex_text(target_line)
                y.append(target_line)

        # Load dataset from news.tsv
        with open(f"{self.path}/news.tsv", newline='', encoding='utf-8') as news_file:
            reader = csv.reader(news_file, delimiter='\t')
            for row in reader:
                # English dataset
                input_line, target_line = row[0].lower().strip(), row[1].lower().strip()
                input_line = regex_text(input_line)
                x.append(input_line)

                # Chinese dataset
                target_line = ' '.join(jieba.cut(target_line))
                target_line = regex_text(target_line)
                y.append(target_line)

        # Split the dataset into train and test
        split_index = int(len(x) * split_ratio)
        self.x_train = x[:split_index]
        self.y_train = y[:split_index]
        self.x_test = x[split_index:]
        self.y_test = y[split_index:]

    def tokenize(self):
        """
        Tokenize the dataset -> sequences -> padding.

        :return: None
        """
        # Tokenize by different dataset.
        self.x_tk.fit_on_texts(self.x_train)
        self.y_tk.fit_on_texts(self.y_train)

        # Set the dataset to sequence.
        self.x_train_seq = self.x_tk.texts_to_sequences(self.x_train)
        self.y_train_seq = self.y_tk.texts_to_sequences(self.y_train)
        self.x_test_seq = self.x_tk.texts_to_sequences(self.x_test)
        self.y_test_seq = self.y_tk.texts_to_sequences(self.y_test)

        # Set the maximum input and output length.
        self.x_len = max(len(seq) for seq in self.x_train_seq)
        self.y_len = max(len(seq) for seq in self.y_train_seq)

        # Padding the sequences to ensure the same size.
        self.x_train_seq = pad_sequences(self.x_train_seq, maxlen=self.x_len, padding='post')
        self.y_train_seq = pad_sequences(self.y_train_seq, maxlen=self.y_len, padding='post')
        self.x_test_seq = pad_sequences(self.x_test_seq, maxlen=self.x_len, padding='post')
        self.y_test_seq = pad_sequences(self.y_test_seq, maxlen=self.y_len, padding='post')

    def get_vocab_size(self) -> tuple[int, int]:
        """
        Return the size of the vocabulary(EN vocab size, ZH vocab size).

        :return: tuple[int, int]
        """
        return len(self.x_tk.word_index) + 1, len(self.y_tk.word_index) + 1

    def create_dataset(self, batch_size):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train_seq, self.y_train_seq))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=len(self.x_train_seq))
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        train_dataset = train_dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]))
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test_seq, self.y_test_seq))
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]))

        return train_dataset, test_dataset


if __name__ == '__main__':
    # Testing
    preprocessor = Preprocessor(path='./data')
    print(preprocessor)