import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from batch_sequence import BatchSequence
import numpy as np
import util
import pickle

class RNN:
    """Recurrent Neural Network model for review prediction
    
    Our model consists of a pretrained embedding layer,
    an RNN layer that processes sequence of words, a Dense
    layer for learning and a softmax output layer. 
    
    Attributes:
        model: Tensorflow RNN model
        mapping: Dictionary of words to integer mapping
        vocab_size: Size of the vocabulary
        sentence_length: Length of a sentence for each features
        epochs: Passes on training data
        batch_size: Number of features in one batch
        learning_rate: Learning rate for our Adam optimizer
    """
    def __init__(self, epoch=20, learning_rate=0.001, batch_size=100, sentence_length=2):
        self.model = None
        self.sentence_length = sentence_length
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
            # Get glove embeddings
            embedding_matrix = util.build_embeddings(self.mapping)
            
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=100,
                input_length=self.sentence_length,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                trainable=False)
                
            rnn_layer = SimpleRNN(1000, activation='relu')
            
            # Create an RNN model
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
        """Perform training on the RNN model
        
        Args:
            data: List of list of words
        """
        self.model = self._get_model()
        # Uncomment this if you want to continue training a saved model
        # self.model = load_model('./RNN/RNN_model')
        self.model.summary()
        sequence = BatchSequence(data,
            self.mapping, self.vocab_size, self.sentence_length, self.batch_size)
        self.hist = self.model.fit(sequence, epochs=self.epochs, verbose=1)
        self.save()

    def save(self):
        self.model.save('./RNN/RNN_model')
        return
    
    def predict(self, data):
        """
        Perform prediction on the RNN model. We return
        number of predictions as well because some features may
        not have enough words and are therefore excluded
        
        Args:
            data: List of list of words
            
        Returns:
            Tuple (accuracy, perplexity, num_of_predictions)
        """
        if (self.model == None):
            self.model = load_model('./{}/RNN_model'.format(self.model_filename))
        perplexity = 0
        accuracy = 0
        count = 0
        
        smoothing = 1 / len(self.mapping)
        
        X_test = []
        Y_test = []
        for line in data:
            if len(line) < self.sentence_length:
                continue
            count = count + 1
            try:
                X_mapping = [self.mapping[word] for word in line[:self.sentence_length]]
                Y_mapping = self.mapping[line[self.sentence_length]]
                X_test.append(X_mapping)
                Y_test.append(Y_mapping)
            except:
                perplexity += 1/0.000001
        X_test = np.array(X_test)
        if (len(X_test) == 0):
            return 0, 0, 0
        model_output = self.model.predict_on_batch(X_test)
        model_output = model_output + smoothing # Smoothing to avoid inf perplexity
        accuracy = np.sum(np.argmax(model_output, axis=1) == Y_test) / count
        # Normalize the output values so they sum up to 1
        # We need to do this because we added smoothing
        normalized = model_output / np.sum(model_output, axis=1)[:,None]
        perplexity = perplexity + np.sum(1 / normalized[np.arange(len(X_test)),Y_test])
        perplexity = perplexity / count
        return accuracy, perplexity, count
