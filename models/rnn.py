import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
import numpy as np
import util
import pickle

class BatchSequence(Sequence):
    """Sequence class for loading training data
    
    Our training labels can become too large to fit in memory since
    each training label is a 1-hot vector of size vocab_size. To save memory,
    we convert the training labels to 1-hot vectors batch-by-batch.
    
    Args:
        data: Training data as a list of words
        mapping: Dictionary of words to integer mapping
        vocab_size: Size of the vocabulary
        sequence_length: Length of a sentence for each features
        batch_size: Number of features in one batch
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
        mapping: Dictionary of words to integer mapping
        vocab_size: Size of the vocabulary
        sequence_length: Length of a sentence for each features
        epochs: Passes on training data
        batch_size: Number of features in one batch
        learning_rate: Learning rate for our Adam optimizer
    """
    def __init__(self, epoch=6, learning_rate=0.001, batch_size=100, sequence_length=50):
        self.model = None
        self.sequence_length = sequence_length
        self.mapping = {}
        self.epochs = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Load vocabulary
        with open('vocab.pkl', 'rb') as f:
            self.mapping = pickle.load(f)
        self.vocab_size = len(self.mapping)
        
    def _get_model(self):
        """Get an instance of the RNN model
        
        Creates a new instance if there isn't one available
        
        Returns:
            Tensorflow RNN model
        """
        if self.model == None:
            # Read glove embeddings
            print("Reading embeddings")
            hits = 0
            misses = 0
            embeddings_index = {}
            with open('glove.6B.50d.txt', encoding="utf8") as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    embeddings_index[word] = coefs
            # Make an embedding matrix for our vocabulary
            print("Building embedding matrix")
            embedding_matrix = np.zeros((len(self.mapping),50))
            for word, i in self.mapping.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    misses += 1
            print("Converted %d words (%d misses)" % (hits, misses))
            
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=50,
                input_length=self.sequence_length,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                trainable=False)
            # (batch_size, sequence_length, features)
            rnn_layer = SimpleRNN(1000, activation='relu')
            self.model = tf.keras.Sequential([
                embedding_layer,
                rnn_layer,
                # batch_size, 1000
                Dense(100, activation='relu'),
                Dense(self.vocab_size, activation='softmax')
            ])
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.model.compile(loss='categorical_crossentropy',
                metrics=['acc'], optimizer=opt)
        return self.model

    def train(self, data):
        """
        Perform training on the RNN model
        
        Args:
            data: List of list of words
        """
        self.model = self._get_model()
        # Uncomment this if you want to continue training a saved model
        # self.model = load_model('./RNN/RNN_model')
        self.model.summary()
        sequence = BatchSequence(data,
            self.mapping, self.vocab_size, self.sequence_length, self.batch_size)
        self.hist = self.model.fit(sequence, epochs=self.epochs, verbose=1)
        self.save()

    def save(self):
        self.model.save('./RNN/RNN_model')
        return
    
    def predict(self, data):
        """
        Perform testing on the RNN model. We return
        number of predictions as well because some features may
        not have enough words and are therefore excluded
        
        Args:
            data: List of list of words
            
        Returns:
            Tuple (accuracy, perplexity, num_of_predictions)
        """
        if (self.model == None):
            self.model = load_model('./RNN/RNN_model')
        perplexity = 0
        accuracy = 0
        
        smoothing = 1 / len(self.mapping)
        
        X_test = []
        Y_test = []
        for line in data:
            if len(line) < self.sequence_length:
                continue
            try:
                X_mapping = [self.mapping[word] for word in line[:self.sequence_length]]
                Y_mapping = self.mapping[line[self.sequence_length]]
                X_test.append(X_mapping)
                Y_test.append(Y_mapping)
            except:
                perplexity += 1/0.000001
        X_test = np.array(X_test)
        print(len(X_test), "features suitable for prediction out of", len(data))
        if (len(X_test) == 0):
            return 0, 0, 0
        model_output = self.model.predict_on_batch(X_test)
        model_output = model_output + smoothing # Smoothing to avoid inf perplexity
        accuracy = np.sum(np.argmax(model_output, axis=1) == Y_test) / len(X_test)
        # Normalize the output values so they sum up to 1
        # We need to do this because we added smoothing
        normalized = model_output / np.sum(model_output, axis=1)[:,None]
        perplexity = perplexity + np.sum(1 / normalized[np.arange(len(X_test)),Y_test])
        perplexity = perplexity / len(X_test)
        return accuracy, perplexity, len(X_test)
