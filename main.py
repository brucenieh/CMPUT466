import raw.readingfiles as readfiles
import pickle, time
from models.ngrams import Ngrams
from models.ANN import ANN
from models.rnn import RNN
from util import initial_setup, evaluate, k_fold, read_data, evaluate_ANN, evaluate_RNN

def tweak_ngram():
    # tweak hyperparameter n for ngram
    performance = []
    for n in range(2,8):
        ngrams_model = Ngrams(n)
        accuracy = evaluate(ngrams_model,
                            data_set[0].training_set,
                            data_set[0].validation_set)
        performance.append([n,accuracy])
    return performance

def tweak_ANN_epoch():
    # tweak hyperparameter epoch for ANN
    performance = []
    epochs = [5, 10, 15, 25, 50]
    for epoch in epochs:
        ANN_model = ANN(epoch=epoch)
        ANN_model.train(data_set[0].training_set)
        accuracy, perplexity = evaluate_ANN(ANN_model,
                                            data_set[0].validation_set,
                                            ANN_model.mapping)
        performance.append([epoch,accuracy,perplexity])
    return performance

def tweak_ANN_lr():
    # tweak hyperparameter learning rate for ANN
    performance = []
    lrs = [0.001, 0.01, 0.1, 0.5, 1, 2]
    for lr in lrs:
        ANN_model = ANN(epoch=5, lr=lr)
        ANN_model.train(data_set[0].training_set)
        accuracy, perplexity = evaluate_ANN(ANN_model,
                                            data_set[0].validation_set,
                                            ANN_model.mapping)
        performance.append([lr,accuracy,perplexity])
    return performance

def tweak_ANN_batch_size():
    # tweak hyperparameter batch size for ANN
    performance = []
    batch_size = [2000, 1500, 1000, 750, 500]
    for bs in batch_size:
        ANN_model = ANN(epoch=5, lr=0.1, batch_size=bs)
        ANN_model.train(data_set[0].training_set)
        accuracy, perplexity = evaluate_ANN(ANN_model,
                                        data_set[0].validation_set,
                                        ANN_model.mapping)
        performance.append([bs,accuracy,perplexity])
    return performance

if __name__ == '__main__':
    initial_setup() # This line needs to be executed only once
    
    training_set = read_data('training_data.csv')
    testing_set = read_data('testing_data.csv')


    # tweak hyper parameter n of n-gram model
    # best_n = (3,0) #n value, accuracy
    # for n in range(2,8):
    #     ngrams_model = Ngrams(n)
    #     accuracy = evaluate(ngrams_model, data_set[0].training_set, data_set[0].validation_set)
    #     print(n,accuracy)
    #     if accuracy > best_n[1]:
    #         best_n = (n,accuracy)
    
    ngrams_model = Ngrams(3)
    accuracy,perplexity = evaluate(ngrams_model, training_set, testing_set, 50)
    print("Ngram accuracy, perplexity: ", accuracy, perplexity)

    ANN_model = ANN(epoch=5, lr=0.1, batch_size=2000)
    ANN_model.train(training_set)
    acc, per = evaluate_ANN(ANN_model,
                            testing_set,
                            ANN_model.mapping)
    print("ANN accuracy, perplexity: ", acc, per)
    
    RNN_model = RNN(epoch=7)
    RNN_model.train(training_set)
    acc, per = evaluate_RNN(RNN_model, testing_set)
    print("RNN accuracy, perplexity: ", acc, per)
