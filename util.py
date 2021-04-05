import csv
import pandas

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

def evaluate(model,training_data,testing_data, sentence_length=50):
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

def main():
    dataset = k_fold(10,'develop.csv')

    for data in dataset:
        print(len(data.validation_set),len(data.training_set))
    
    # print(dataset[0].validation_set[3:7])
        

if __name__ == '__main__':
    main()
