import csv
import pandas
import pickle
import numpy as np
import raw.readingfiles as readfiles
import nltk

class Training_data():
    """Data structure to store the training data
    
    Attributes:
        validation_set: list of list of words as validation set
        training_set: list of list of words as training set
    """
    def __init__(self):
        self.validation_set = None
        self.training_set = None

def read_data(path):
    """Return dataset as a list of list of words
    
    Args:
        path (str): path to the data file in csv format
    
    Returns:
        list: a list of list of words
    """
    # Read csv file, convert 'text' column from string to list
    df = pandas.read_csv(path, converters={'text': eval})
    # Convert the Series object to a python list
    data = df['text'].tolist()
    return data

def k_fold(k,path):
    """Split dataset into k folds for cross validation
    
    Args:
        k (int): number of folds to split into
        path (str): path to the data file in csv format
    
    Returns:
        list: a list of k number of Training_data objects
    """
    dataset = []
    # Read csv file, convert 'text' column from string to list
    df = pandas.read_csv(path, converters={'text': eval})
    # Convert the Series object to a python list
    data = df['text'].tolist()
    chunk_size = len(data)//k
    for i in range(k):
        td = Training_data()
        td.validation_set = data[i*chunk_size:(i+1)*chunk_size]
        td.training_set = data[0:i*chunk_size] + data[(i+1)*chunk_size:-1]
        dataset.append(td)
    return dataset

def evaluate_RNN(model,testing_data):
    """Returns accuracy and perplexity scores for RNN
    
    Args:
        model: an instance of the RNN class
        testing_data: A list of test features returned by the read_data function
    
    Returns:
        (float, float): accuracy and perplexity scores
    """
    accuracy = 0
    perplexity = 0
    size = 0
    
    for i in range(0, len(testing_data), model.batch_size):
        acc, perp, n = model.predict(testing_data[i:i + model.batch_size])
        accuracy = accuracy + acc * n
        perplexity = perplexity + perp * n
        size = size + n
    
    return accuracy / size, perplexity / size

def build_vocab(data,name):
    """Builds a word-to-integer mapping for the given dataset
    
    Args:
        data: list of list of words
        name: Name of the pickle file where the mapping is dumped
    
    Returns:
        dict: a word-to-integer mapping for the given dataset
    """
    mapping = {}
    counter = 0
    for line in data:
        for word in line:
            if word in mapping:
                continue
            else:
                mapping[word] = counter
                counter += 1
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)
    return mapping

def build_embeddings(vocab):
    """Builds an embedding matrix using GloVe embeddings 

    Args:
        vocab (dict): A vocabulary mapping built with build_vocab
    
    Returns:
        np.array: A numpy matrix to be used as weight matrix
                  by the embedding layer
    """
    print("Reading embeddings")
    hits = 0
    misses = 0
    embeddings_index = {}
    with open('glove.6B.100d.txt', encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    # Make an embedding matrix for our vocabulary
    print("Building embedding matrix")
    embedding_matrix = np.zeros((len(vocab),100))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix

def initial_setup():
    print("Downloading NLTK packages")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    print("Building training and testing dataset")
    readfiles.make_datasets()
    print("Building vocabulary")
    training_set = read_data('training_data.csv')
    build_vocab(training_set, 'vocab')

def main():
    dataset = k_fold(10,'develop.csv')

    for data in dataset:
        print(len(data.validation_set),len(data.training_set))
    
    # print(dataset[0].validation_set[3:7])
        

if __name__ == '__main__':
    main()
