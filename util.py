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

def evaluate(model,training_data,testing_data):
    """returns the accuracy and perplexity of the model

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
        sentence = test[0:-2]
        target = test[-2]
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

def build_vocab(data,name):
    """Build model's vocabulary from every word in training data and save to a
    pickle file.

    Args:
        data (list of str): training data containing reviews
        name (str): filename for pickle
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

def main():
    dataset = k_fold(10,'develop.csv')

    for data in dataset:
        print(len(data.validation_set),len(data.training_set))
    
    # print(dataset[0].validation_set[3:7])
        

if __name__ == '__main__':
    main()
