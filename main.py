import raw.readingfiles as readfiles
from models.rnn import RNN
from util import k_fold,read_data, evaluate_RNN
import util

if __name__ == '__main__':
    # util.initial_setup() # This line needs to be executed only once
    
    readfiles.make_datasets()
    exit()
    data_set = k_fold(10, 'training_data.csv')
    training_set = read_data('training_data.csv')
    testing_set = read_data('testing_data.csv')

    util.build_vocab(training_set, 'vocab')
    RNN_model = RNN(epoch=2)
    RNN_model.train(training_set)
    print(evaluate_RNN(RNN_model, testing_set))
