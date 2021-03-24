import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras import Sequential
import numpy as np

class RNN:
    """Recurrent Neural Network model for review prediction
    
    Attributes:
        model: Tensorflow RNN model
    """
    def __init__(self):
        self.model = None
        self.sequence_length = 50
        self.mapping = {}
        self.vocab_size = None
        pass
    
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
            rnn_layer = SimpleRNN(1000, activation='relu')
            self.model = tf.keras.Sequential([
                embedding_layer,
                rnn_layer,
                Dense(100, activation='relu'),
                Dense(100, activation='relu'),
                Dense(self.vocab_size, activation='softmax')
            ])
        return self.model

    def _get_x_y(self, data):
        """
        Transform data from pandas DataFrame to features and labels
        
        Args:
            data: Pandas DataFrame containing the dataset
        
        Returns:
            (features, labels) tuple containing the feature set
            and the corresponding label set
        """
        # Generate vocabulary. Code written by Kean
        counter = 0
        for line in data:
            for word in line:
                if word in self.mapping:
                    continue
                else:
                    self.mapping[word] = counter
                    counter += 1
        self.vocab_size = len(self.mapping)
        X_train = []
        Y_train = []
        for line in data:
            if len(line) < self.sequence_length + 1:
                continue
            X_train.append([self.mapping[word] for word in line[:self.sequence_length]])
            new_y = np.zeros(self.vocab_size)
            new_y[self.mapping[line[self.sequence_length]]] = 1
            Y_train.append(new_y)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train
    
    def train(self, data):
        """
        Perform training on the RNN model
        
        Args:
            data: Pandas DataFrame containing the train dataset
        """
        X_train, Y_train = self._get_x_y(data)
        model = self._get_model()
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
        model.summary()
        model.fit(X_train, Y_train, epochs=100, verbose=1)
        self.model = model
    
    def predict(self, data):
        """
        Perform testing on the RNN model
        
        Args:
            data: Pandas DataFrame containing the test dataset
            
        Returns:
            Prediction accuracy on test data
        """
        pass
        
