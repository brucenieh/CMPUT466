import pickle, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import Sequence
import os

class BatchSequence(Sequence):
    """Sequence class for loading training data
    
    Our training labels can become too large to fit in memory since
    each training label is a 1-hot vector of size vocab_size. To fix this,
    we train our model in batches and generate the labels for batches
    on the fly
    """
    def __init__(self, data, mapping, vocab_size, sequence_length, batch_size):
        self.data = [i for i in data if len(i) > sequence_length]
        self.mapping = mapping
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        X_train = []
        Y_train = []
        for line in self.data[idx * self.batch_size:(idx + 1) * self.batch_size]:
            X_train.append([self.mapping[word] for word in line[:self.sequence_length]])
            new_y = np.zeros(self.vocab_size)
            new_y[self.mapping[line[self.sequence_length]]] = 1
            Y_train.append(new_y)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train

class ANN:
    def __init__(self, sentence_length=50, batch_size=1000, epoch=10, lr=0.001):
        """Our ANN model is a simple feed-forward neural network and is based on
        Keras' Sequential model. Mapping is the model's vocabulary, which is a
        dictionary with key-value, 'word': incrementing index. Mapping is 
        indexed based on training data. Example:
        self.mapping = {
            'this': 0,
            'is' : 1,
            'all': 2,
            'vocab': 3
        }

        Args:
            sentence_length (int): sentence length to predict
            batch_size (int): size of sentences in each data batch for training
            epoch (int): epochs to train our model
            lr (float): learning rate for our model
        """
        # load our pre-built vocabulary mapping from a pickle file
        with open('vocab.pkl', 'rb') as f:
            self.mapping = pickle.load(f)
        
        self.vocab = len(self.mapping)
        self.epoch = epoch
        self.batch_size = batch_size
        self.sentence_length = sentence_length

        # initialize Sequential layer and add layers
        self.model = Sequential()
        self.model.add(Embedding(self.vocab, 100, input_length=sentence_length,
                       trainable=True))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(self.vocab, activation='softmax'))

        # initialize optimizer to use specified learning rate
        opt = Adam(learning_rate=lr)
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'],
                           optimizer=opt)
        self.model.summary()

    def data_reader(self, data, batch_size):
        """Returns a generator to generate data batches from the whole dataset
        based on batch_size.

        Args:
            data (list of str): our whole dataset of sentences
            batch_size (int): size of sentences in each data batch

        Yields:
            list of str: data batch
        """
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def train(self, data):
        """Preprocesses training data and trains our model by batch. Our X_train
        is a Numpy array of size (batch_size, 50), containing the first 50 words
        of each review (represented as index of mapping). Our Y_train is a Numpy
        array of size (batch_size, vocab_size), containing the 51st word of each
        review (represented as one-hot vector of the whole vocab). 

        Args:
            data (list of str): training data containing movie reviews
        """
        dr = self.data_reader(data, self.batch_size)
        # iterate each batch of data
        for batch_data in dr:
            X_train = []
            Y_train = []
            # iterate each review in batch of data
            for line in batch_data:
                # ignore reviews shorter than 51
                if len(line) < self.sentence_length + 1:
                    continue
                # select first 50 word and convert to index in vocab mapping
                sentence = line[:self.sentence_length]
                sentence = np.array([self.mapping[word] for word in sentence]) 
                # initialize our one-hot vector and marking the correct word as 1
                y_word = line[self.sentence_length]
                y_vector = np.zeros(self.vocab)
                y_vector[self.mapping[y_word]] = 1
                
                X_train.append(sentence)
                Y_train.append(y_vector)
            # convert list to Numpy array
            X_train = np.asarray(X_train)
            Y_train = np.asarray(Y_train)

            # train on this batch of data for epoch times
            for _ in range(self.epoch):
                self.model.train_on_batch(X_train, Y_train)

    def save(self):
        """Saves the model locally to the specified directory to save time from
        retraining whole model.
        """
        self.model.save('./ANN/ANN_model')
        return        

    def predict(self, text):
        """Preprocess testing data and predicts the 51st word. When we see an
        unknown word (words that are not in vocab), we assign it the index of
        the previous word. If unknown word is at the start of a sentence, we
        randomly assign an index to it.

        Args:
            text (str): sentence from testing data to predict next word for

        Returns:
            Numpy array: probablity distribution of all the words in vocab for
                each review, size of (batch_size, vocab_size) 
        """
        X_test = []
        # iterate each review
        for line in text:
            sentence = []
            # iterate each word in a review
            for i in range(self.sentence_length):
                # try looking up the word in vocab mapping
                try:
                    sentence.append(self.mapping[line[i]])
                # if word does not exist in mapping
                except KeyError:
                    # if first word is unknown, assign random index, else copy
                    # index of previous word
                    if len(sentence) == 0:
                        sentence.append(random.randint(0, self.vocab - 1))
                    else:    
                        sentence.append(sentence[-1])
            X_test.append(sentence)
        # convert list to Numpy array
        X_test = np.asarray(X_test)
        Y_hat = self.model.predict(X_test)

        return Y_hat
