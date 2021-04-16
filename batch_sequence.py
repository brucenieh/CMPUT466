from tensorflow.keras.utils import Sequence
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