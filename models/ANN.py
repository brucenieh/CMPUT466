import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class ANN:
    def __init__(self):
        """Our ANN model is a simple feed-forward neural network and is based on
        Keras' Sequential model.
        """
        self.model = Sequential()

    def train(self, data):
        """Preprocesses training data and trains our model.

        Args:
            data (list of str): training data
        """
        pass

    def predict(self, text):
        """Predicts the next word based on our model.

        Args:
            text (str): sentence from testing data to predict next word for
        """
        pass