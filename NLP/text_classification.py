import json
import tensorflow as tf
import csv
import random
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class NLPClassify:
    def __init__(self):
        self.batch_size = 128
        self.embedding_dim = 100
        self.oov_tok = "<OOV>"
        self.sentences, self.labels = self.load_text_from_csv("../Data/training_cleaned.csv")

    @staticmethod
    def load_text_from_csv(filename, delimiter=','):
        """
        Load text from csv and extract sentences and labels
        """
        num_sentences = 0
        corpus = []
        sentences = []
        labels = []

        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for row in reader:
                list_item = []
                list_item.append(row[5])
                this_label = row[0]
                if this_label == '0':
                    list_item.append(0)
                else:
                    list_item.append(1)
                num_sentences = num_sentences + 1
                corpus.append(list_item)

        random.shuffle(corpus)
        # 1/5 data will be used
        for x in range(num_sentences // 5):
            sentences.append(corpus[x][0])
            labels.append(corpus[x][1])

        return sentences, labels

    def tokenize(self, trunc_type='post', padding_type='post'):
        """
        Tokenize text and pad to the maximum length
        """

        self.max_length = 16  # max([len(sentence) for sentence in self.sentences])
        tokenizer = Tokenizer(oov_token=self.oov_tok)
        tokenizer.fit_on_texts(self.sentences)
        self.word_index = tokenizer.word_index
        self.vocab_size = len(self.word_index)

        sequences = tokenizer.texts_to_sequences(self.sentences)
        self.padded_sentences = pad_sequences(sequences,
                                              maxlen=self.max_length,
                                              padding=padding_type,
                                              truncating=trunc_type)

    def split_dataset(self, test_portion=0.1, ):
        training_size = len(self.padded_sentences)
        split = int(test_portion * training_size)

        test_sequences = self.padded_sentences[0:split]
        training_sequences = self.padded_sentences[split:training_size]
        test_labels = self.labels[0:split]
        training_labels = self.labels[split:training_size]

        train_ds = tf.data.Dataset.from_tensor_slices((training_sequences, training_labels)).shuffle(1000).batch(
            self.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(self.batch_size)

        return train_ds, test_ds

    def load_external_embedding(self, filename):
        embeddings_index = {}
        with open(filename) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        self.embeddings_matrix = np.zeros((self.vocab_size + 1, self.embedding_dim))

        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embeddings_matrix[i] = embedding_vector

    def build_model(self):
        self.model = tf.keras.Sequential([
            Embedding(self.vocab_size + 1,
                      self.embedding_dim,
                      input_length=self.max_length,
                      weights=[self.embeddings_matrix],
                      trainable=False),

            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),
            LSTM(64),
            Dense(self.vocab_size // 2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self, train_ds, test_ds, save_model='True'):
        callbacks = myCallback()
        history = self.model.fit(train_ds,
                                 epochs=200,
                                 validation_data=test_ds,
                                 callbacks=[callbacks]
                                 )
        if save_model:
            self.model.save("text_binary_classify.h5")
            print(f"Model saved")

        return history


class myCallback(tf.keras.callbacks.Callback):
    """
    Define a Callback class that stops training once accuracy reaches 97.0%
    """

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))

    nlp_binary = NLPClassify()
    nlp_binary.tokenize()
    train_ds, test_ds = nlp_binary.split_dataset()
    nlp_binary.load_external_embedding('../Data/glove.6B.100d.txt')
    nlp_binary.build_model()
    nlp_binary.train_model(train_ds, test_ds)
