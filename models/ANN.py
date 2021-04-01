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
        Keras' Sequential model.

        Args:
            sentence_length (int): sentence length to predict
            batch_size (int): size of sentences in each data batch for training
            epoch (int): epochs to train our model
            lr (float): learning rate for our model
        """
        # load our pre-built vocabulary mapping from a previously saved pickle
        with open('vocab.pkl', 'rb') as f:
            self.mapping = pickle.load(f)
        self.vocab = len(self.mapping)
        self.epoch = epoch
        self.batch_size = 100

        self.sentence_length = sentence_length
        self.model = Sequential()
        self.model.add(Embedding(self.vocab, 100, input_length=sentence_length,
                       trainable=True))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(self.vocab, activation='softmax'))

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
        """Preprocesses training data and trains our model.

        Args:
            data (list of str): training data containing movie reviews
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

        dr = self.data_reader(data, self.batch_size)
        for batch_data in dr:
            X_train = []
            Y_train = []
            for line in batch_data:
                if len(line) < self.sentence_length + 1:
                    continue
                vector = np.zeros(self.vocab)
                y_word = line[self.sentence_length]
                vector[self.mapping[y_word]] = 1
                # for word in line:
                #     index = mapping[word]
                #     vector[index] += 1
                sentence = line[:self.sentence_length]
                sentence = np.array([self.mapping[word] for word in sentence])
                
                X_train.append(sentence)
                Y_train.append(vector)
            X_train = np.asarray(X_train)
            Y_train = np.asarray(Y_train)
            # Create a callback that saves the model's weights
            # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ANN/',
            #                                                  save_weights_only=True,
            #                                                  verbose=1)
            for _ in range(self.epoch):
                self.model.train_on_batch(X_train, Y_train)

    def save(self):
        self.model.save('./ANN/ANN_model')
        return
        #     print(vector)
        # print(mapping)
        # vector = vectorizer.transform([' '.join(data[0])])
        # print(vector.shape)
        # training_data
        # with open('ANN.pkl', 'wb') as f:
        #     pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL) 
        
        
    # @tf.function(experimental_relax_shapes=True)
    def predict(self, text):
        """Predicts the next word based on our model.

        Args:
            text (str): sentence from testing data to predict next word for
        """
        # self.model = load_model('./ANN/ANN_model')
        # if len(text) < self.sentence_length:
        #     print("Sentence too short!")
        #     return 0
        # try:
        #     sentence = [self.mapping[word] for word in text]
        # except:
        #     return []
        # sentence = np.asarray(sentence).reshape((1, self.sentence_length))
        # # tf.compat.v1.disable_eager_execution()
        # yhat = self.model.predict(sentence)
        X_test = []
        for line in text:
            sentence = []
            for i in range(self.sentence_length):
                try:
                    sentence.append(self.mapping[line[i]])
                except KeyError:
                    if len(sentence) == 0:
                        sentence.append(random.randint(0, self.vocab - 1))
                    else:    
                        sentence.append(sentence[-1])
            X_test.append(sentence)

        X_test = np.asarray(X_test)
        yhat = self.model.predict(X_test)

        return yhat
