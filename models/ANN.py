import pickle
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
    def __init__(self, sentence_length=50, epoch=10, lr=0.001):
        """Our ANN model is a simple feed-forward neural network and is based on
        Keras' Sequential model.
        """
        with open('vocab.pkl', 'rb') as f:
            self.mapping = pickle.load(f)
        vocab = len(self.mapping)
        self.epoch = epoch
        self.batch_size = 100

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
        
        sequence = BatchSequence(data,
            self.mapping, len(self.mapping), self.sentence_length, self.batch_size)
        
        
        # Create a callback that saves the model's weights
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ANN/',
        #                                                  save_weights_only=True,
        #                                                  verbose=1)
        self.model.fit(sequence, epochs=self.epoch, verbose=1)
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
