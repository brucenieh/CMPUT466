import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer

class ANN:
    def __init__(self):
        """Our ANN model is a simple feed-forward neural network and is based on
        Keras' Sequential model.
        """
        self.model = Sequential()

    def train(self, data, sentence_length=30):
        """Preprocesses training data and trains our model.

        Args:
            data (list of str): training data
        """
        mapping = {}
        counter = 0
        for line in data:
            for word in line:
                if word in mapping:
                    continue
                else:
                    mapping[word] = counter
                    counter += 1
        
        vocab = len(mapping)
        # print(vocab)

        # self.model.add(Embedding(vocab, 20, input_length=sentence_length, trainable=True))
        # self.model.add(Dense(10000, activation="relu"))
        # self.model.add(Dense(1000, activation="softmax"))

        # self.model.add(Embedding(vocab, 50, input_length=30, trainable=True))
        self.model.add(Dense(1000, activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(1, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

        X_train = np.ndarray([1, sentence_length + 1])
        for line in data:
            # vector = np.zeros(len(mapping))
            # for word in line:
            #     index = mapping[word]
            #     vector[index] += 1
            sentence = np.array([mapping[word] for word in line])
            if len(sentence) < sentence_length + 1:
                continue
            X_train = np.vstack((X_train, sentence[:sentence_length + 1]))
        print(X_train[:,:sentence_length].shape)
        print(X_train[:,sentence_length:sentence_length + 1].shape)
        self.model.fit(X_train[:,:sentence_length], X_train[:,sentence_length:sentence_length + 1], epochs=10, verbose=2)
        #     print(vector)
        # print(mapping)
        # vector = vectorizer.transform([' '.join(data[0])])
        # print(vector.shape)
        # training_data 
        pass

    def predict(self, text):
        """Predicts the next word based on our model.

        Args:
            text (str): sentence from testing data to predict next word for
        """
        pass