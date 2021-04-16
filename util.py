import csv
import pandas
import pickle
import numpy as np
import nltk

import raw.readingfiles as readfiles
from models.ngrams import Ngrams
from models.ANN import ANN

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

def evaluate_ngrams(model,training_data,testing_data, sentence_length=50):
    """Returns the accuracy and perplexity of the model

    Args:
        model (Model): Model to be evaluated
        training_data (Training_data): Data for the model to be trained on
        testing_data (list(list(str))): A list of sentences

    Returns:
        float: Accuracy of the model
        float: Perplexity of the model
    """
    accuracy = 0
    perplexity = 0
    try:
        model.train(training_data)
    except Exception as e:
        print('model training failed\n', e)
        exit()
    
    for test in testing_data:
        if len(test) < sentence_length + 1:
            continue
        sentence = test[0:sentence_length]
        target = test[sentence_length]
        try:
            prediction,prob_distrib = model.predict(sentence)
        except Exception as e:
            print('model predicting failed\n', e)
            exit()
        
        # Update accuracy
        if prediction[0] == target:
            accuracy += 1
        
        # Update perplexity
        if prob_distrib:
            if target in prob_distrib.keys():
                perplexity += 1/prob_distrib[target]
            else:
                perplexity += 1/prob_distrib['<UNK>']
        else:
            perplexity += 1/0.000001
    
    accuracy = accuracy/len(testing_data)
    perplexity = perplexity/len(testing_data)

    return accuracy,perplexity

def evaluate_ANN(model,testing_data,mapping):
    """Returns the accuracy and perplexity of the ANN model

    Args:
        model (Model): ANN Model
        testing_data (list of str): Data for the model to be tested on
        mapping (dictionary): Vocabulary of model

    Returns:
        float: Accuracy of the model
        float: Perplexity of the model
    """
    accuracy = 0
    perplexity = 0
    filtered_data = []
    targets = []
    for test in testing_data:
        # skip sentences shorter than 51 words
        if len(test) < model.sentence_length + 1:
            continue
        filtered_data.append(test[:model.sentence_length])
        targets.append(test[model.sentence_length])
    # predict using filtered data
    filtered_data = np.asarray(filtered_data)
    prob_distrib = model.predict(filtered_data)

    for i in range(len(prob_distrib)):
        target = targets[i]
        # check if word is correct
        try:
            correct_index = mapping[target]
        except:
            perplexity += 1/0.000001

        # compare prediction from model and update accuracy
        if np.argmax(prob_distrib[i]) == correct_index:
            accuracy += 1
        # update perplexity
        try:
            perplexity += 1/prob_distrib[i][correct_index]
        except:
            perplexity += 1/0.000001

    accuracy = accuracy/len(filtered_data)
    perplexity = perplexity/len(filtered_data)

    return accuracy,perplexity

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
            # for each unseen word, we index it incrementally
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

def tweak_ngram(data):
    """Tweak hyperparameter n for ngrams model

    Args:
        data (Training_data): class containing training data and validation
                              data for each k-fold

    Returns:
        list: each index contains hyperparameter n, accuracy and perplexity
    """
    performance = []
    for n in range(2,8):
        ngrams_model = Ngrams(n)
        accuracy, perplexity = evaluate_ngrams(ngrams_model,
                                        data.training_set,
                                        data.validation_set)
        performance.append([n, accuracy, perplexity])
    return performance

def tweak_ANN_epoch(data):
    """Tweak hyperparameter epoch for ANN model

    Args:
        data (Training_data): class containing training data and validation
                              data for each k-fold

    Returns:
        list: each index contains hyperparameter epoch, accuracy and perplexity
    """
    performance = []
    epochs = [5, 10, 15, 25, 50]
    for epoch in epochs:
        ANN_model = ANN(epoch=epoch)
        ANN_model.train(data.training_set)
        accuracy, perplexity = evaluate_ANN(ANN_model,
                                            data.validation_set,
                                            ANN_model.mapping)
        performance.append([epoch, accuracy, perplexity])
    return performance

def tweak_ANN_lr(data):
    """Tweak hyperparameter learning rate for ANN model

    Args:
        data (Training_data): class containing training data and validation
                              data for each k-fold

    Returns:
        list: each index contains hyperparameter learning rate, accuracy and
              perplexity
    """
    performance = []
    lrs = [0.001, 0.01, 0.1, 0.5, 1, 2]
    for lr in lrs:
        ANN_model = ANN(epoch=5, lr=lr)
        ANN_model.train(data.training_set)
        accuracy, perplexity = evaluate_ANN(ANN_model,
                                            data.validation_set,
                                            ANN_model.mapping)
        performance.append([lr, accuracy, perplexity])
    return performance

def tweak_ANN_batch_size(data):
    """Tweak hyperparameter batch size for ngrams model

    Args:
        data (Training_data): class containing training data and validation
                              data for each k-fold

    Returns:
        list: each index contains hyperparameter batch size, accuracy and
              perplexity
    """
    performance = []
    batch_size = [2000, 1500, 1000, 750, 500]
    for bs in batch_size:
        ANN_model = ANN(epoch=5, lr=0.1, batch_size=bs)
        ANN_model.train(data.training_set)
        accuracy, perplexity = evaluate_ANN(ANN_model,
                                            data.validation_set,
                                            ANN_model.mapping)
        performance.append([bs, accuracy, perplexity])
    return performance

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
