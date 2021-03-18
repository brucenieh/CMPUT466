import tensorflow as tf

class RNN:
    """Recurrent Neural Network model for review prediction
    
    Attributes:
        model: Tensorflow RNN model
    """
    def __init__(self):
        self.model = None
        pass
    
    def _get_model(self, data=None):
        """Get an instance of the RNN model
        
        Creates a new instance if there isn't one available
        
        Args:
            data: List of strings for generating vocabulary
        
        Returns:
            Tensorflow RNN model
        """
        if self.model == None:
            # TODO: Decide TextVectorization layer arguments
            vectorize_layer = TextVectorization()
            vectorize_layer.adapt(data)
            # TODO: Determine layer size
            # TODO: Determine layers
            # TODO: Determine Embedding layer argumenets
            # TODO: Determine output layer arguments
            self.model = tf.keras.Sequential([
                vectorize_layer,
                tf.keras.layers.Embedding()
                tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
                tf.keras.layers.Dense(64, activation='relu')
                tf.keras.layers.Dense(len(vectorize_layer.get_vocabulary()))
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
        # x should be the 'text' field minus the last word
        # y should be the last word
        pass
    
    def train(self, data):
        """
        Perform training on the RNN model
        
        Args:
            data: Pandas DataFrame containing the train dataset
        """
        x_train, y_train = self._get_x_y(data)
        model = self._get_model(data['text'].tolist())
        model.compile(metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=100)
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
        
