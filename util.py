import csv
import pandas
import pickle
import numpy as np

class Training_data():
    """data structure to store the training data
    validation_set: list of list of words as validation set
    training_set: list of list of words as training set
    """
    def __init__(self):
        self.validation_set = None
        self.training_set = None

def read_data(path):
    """return dataset as a list of
    Args:
        path (str): path to the data file in csv format
    """
    # Read csv file, convert 'text' column from string to list
    df = pandas.read_csv(path, converters={'text': eval})
    # Convert the Series object to a python list
    data = df['text'].tolist()
    return data

def k_fold(k,path):
    """split dataset into k folds for cross validation
    Args:
        k (int): number of fold to split into
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
    accuracy = 0
    perplexity = 0
    size = 0
    
    for i in range(0, len(testing_data), model.batch_size):
        print("Batch", i / model.batch_size, "of", np.ceil(len(testing_data) / model.batch_size))
        acc, perp, n = model.predict(testing_data[i:i + model.batch_size])
        accuracy = accuracy + acc * n
        perplexity = perplexity + perp * n
        size = size + n
    
    return accuracy / size, perplexity / size

def build_vocab(data,name):
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

def main():
    dataset = k_fold(10,'develop.csv')

    for data in dataset:
        print(len(data.validation_set),len(data.training_set))
    
    # print(dataset[0].validation_set[3:7])
        

if __name__ == '__main__':
    main()
