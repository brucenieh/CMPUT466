import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
import os

class ANN:
    def __init__(self, sentence_length=50, epoch=10, lr=0.001):
        """Our ANN model is a simple feed-forward neural network and is based on
        Keras' Sequential model.
        """
        with open('vocab.pkl', 'rb') as f:
            self.mapping = pickle.load(f)
        vocab = len(self.mapping)
        self.epoch = epoch

        self.sentence_length = sentence_length
        self.model = Sequential()
        self.model.add(Embedding(vocab, 100, input_length=sentence_length, trainable=True))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(vocab, activation='softmax'))

        opt = Adam(learning_rate=lr)
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)
        self.model.summary()

    def train(self, data):
        """Preprocesses training data and trains our model.

        Args:
            data (list of str): training data
        """
        # mapping = {}
        # counter = 0
        # for line in data:
        #     for word in line:
        #         if word in mapping:
        #             continue
        #         else:
        #             mapping[word] = counter
        #             counter += 1
        # print("###################\n",os.getcwd())
        # with open('vocab.pkl', 'rb') as f:
        #     mapping = pickle.load(f)

        # vocab = len(mapping)
        # print(vocab)

        # self.model.add(Embedding(vocab, 20, input_length=sentence_length, trainable=True))
        # self.model.add(Dense(10000, activation="relu"))
        # self.model.add(Dense(1000, activation="softmax"))

        
        # X_train = np.ndarray((1, sentence_length + 1))
        # Y_train = np.ndarray((1, len(mapping)))
        X_train = []
        Y_train = []
        for line in data:
            if len(line) < self.sentence_length + 1:
                continue
            vector = np.zeros(len(self.mapping))
            y_word = line[self.sentence_length]
            vector[self.mapping[y_word]] = 1
            # for word in line:
            #     index = mapping[word]
            #     vector[index] += 1
            sentence = np.array([self.mapping[word] for word in line])
            
            X_train.append(sentence[:self.sentence_length])
            Y_train.append(vector)
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        print(X_train.shape)
        print(Y_train.shape)
        # Create a callback that saves the model's weights
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ANN/',
        #                                                  save_weights_only=True,
        #                                                  verbose=1)
        self.model.fit(X_train, Y_train, epochs=self.epoch, verbose=1)
        self.model.save('./ANN/ANN_model')
        #     print(vector)
        # print(mapping)
        # vector = vectorizer.transform([' '.join(data[0])])
        # print(vector.shape)
        # training_data
        # with open('ANN.pkl', 'wb') as f:
        #     pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL) 
        
        return
    # @tf.function(experimental_relax_shapes=True)
    def predict(self, text):
        """Predicts the next word based on our model.

        Args:
            text (str): sentence from testing data to predict next word for
        """
        self.model = load_model('./ANN/ANN_model')
        if len(text) < self.sentence_length:
            print("Sentence too short!")
            return 0
        try:
            sentence = [self.mapping[word] for word in text]
        except:
            return []
        sentence = np.asarray(sentence).reshape((1, self.sentence_length))
        # tf.compat.v1.disable_eager_execution()
        yhat = self.model.predict(sentence)

        return yhat.reshape(-1)
