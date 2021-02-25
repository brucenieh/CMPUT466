import pandas
import ast

class Training_data():
    """data structure to store the training data
    validation_set: list of list of words as validation set
    training_set: list of list of words as training set
    """
    def __init__(self):
        self.validation_set = None
        self.training_set = None

def k_fold(k,path):
    """split dataset into k folds for cross validation

    Args:
        k (int): number of fold to split into
        path (str): path to the data file in csv format

    Returns:
        list: a list of k number of Training_data objects
    """
    dataset = []
    data = []
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


def main():
    dataset = k_fold(10,'develop.csv')

    for data in dataset:
        print(len(data.validation_set),len(data.training_set))
    
    # print(dataset[0].validation_set[3:7])
        

if __name__ == '__main__':
    main()
