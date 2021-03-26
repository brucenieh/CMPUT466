import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import Sequence
import numpy as np
import util

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

class RNN:
    """Recurrent Neural Network model for review prediction
    
    Attributes:
        model: Tensorflow RNN model
    """
    def __init__(self, sequence_length=50):
        self.model = None
        self.sequence_length = sequence_length
        self.mapping = {}
        self.inverse_mapping = {}
        self.epochs = 100
        self.vocab_size = None
        pass
        
    def mygenerator(self, data):
        for line in data:
            X_train = np.array(line[:-1])
            Y_train = np.zeros(self.vocab_size)
            Y_train[line[-1]] = 1
            yield X_train, Y_train

    def _get_model(self):
        """Get an instance of the RNN model
        
        Creates a new instance if there isn't one available
        
        Returns:
            Tensorflow RNN model
        """
        if self.model == None:
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=100,
                input_length=self.sequence_length,
                trainable=True)
            # (batch_size, sequence_length, features)
            rnn_layer = SimpleRNN(1000, activation='relu')
            self.model = tf.keras.Sequential([
                embedding_layer,
                rnn_layer,
                # batch_size, 1000
                Dense(100, activation='relu'),
                Dense(100, activation='relu'),
                Dense(self.vocab_size, activation='softmax')
            ])
            self.model.compile(loss='categorical_crossentropy',
                metrics=['acc'], optimizer='adam')
        return self.model

    def train(self, data):
        """
        Perform training on the RNN model
        
        Args:
            data: Pandas DataFrame containing the train dataset
        """
        self.mapping = util.build_vocab(data, 'vocab')
        self.vocab_size = len(self.mapping)
        model = self._get_model()
        model.summary()
        sequence = BatchSequence(data,
            self.mapping, self.vocab_size, self.sequence_length, 100)
        self.hist = model.fit(sequence, epochs=self.epochs, verbose=1)
        self.model = model
        self.save()

    def save(self):
        self.model.save('./RNN/RNN_model')
        return
    
    def predict(self, text):
        """
        Perform testing on the RNN model
        
        Args:
            data: Pandas DataFrame containing the test dataset
            
        Returns:
            Prediction accuracy on test data
        """
        self.model = load_model('./RNN/RNN_model')
        if len(text) < self.sequence_length:
            print("Sentence too short!")
            return 0
        try:
            sentence = [self.mapping[word] for word in text]
        except:
            return []
        sentence = np.asarray(sentence).reshape((1, self.sequence_length))
        yhat = self.model.predict(sentence)
        return self.inverse_mapping[np.argmax(yhat)]
