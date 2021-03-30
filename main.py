import raw.readingfiles as readfiles
from models.rnn import RNN
from util import evaluate, k_fold,read_data, evaluate_RNN
import util

if __name__ == '__main__':
    # readfiles.make_datasets()
    data_set = k_fold(10, 'training_data.csv')
    training_set = read_data('training_data.csv')
    testing_set = read_data('testing_data.csv')


    # ngrams_model.train(data_set[0].training_set)
    # ngrams_model.train(x)
    # print(evaluate(ngrams_model, data_set[0].training_set, data_set[0].validation_set))

    # tweak hyper parameter n of n-gram model
    # best_n = (3,0) #n value, accuracy
    # for n in range(2,8):
    #     ngrams_model = Ngrams(n)
    #     accuracy = evaluate(ngrams_model, data_set[0].training_set, data_set[0].validation_set)
    #     print(n,accuracy)
    #     if accuracy > best_n[1]:
    #         best_n = (n,accuracy)
    
    # ngrams_model = Ngrams(3)
    # accuracy,perplexity = evaluate(ngrams_model, training_set, testing_set)
    # print(accuracy,perplexity)

    util.build_vocab(training_set, 'vocab')
    RNN_model = RNN()
    # print(training_set[:5])
    RNN_model.train(training_set)
    print(evaluate_RNN(RNN_model, testing_set))
    # ANN_model.train(["abc"])
