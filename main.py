import raw.readingfiles as readfiles
import pickle, time
from models.ngrams import Ngrams
from models.ANN import ANN
from util import build_vocab, evaluate, k_fold, read_data, evaluate_ANN

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
    # readfiles.make_datasets()
    data_set = k_fold(10, 'training_data.csv')
    # training_set = read_data('training_data.csv')
    # testing_set = read_data('testing_data.csv')

    accuracy = []
    perplexity = []
    start = time.time()
    i = int(input())
    build_vocab(data_set[i].training_set, 'vocab')
    ANN_model = ANN(epoch=5, lr=0.1, batch_size=2000)
    ANN_model.train(data_set[i].training_set)
    acc, per = evaluate_ANN(ANN_model,
                            data_set[i].validation_set,
                            ANN_model.mapping)
    print("time", time.time() - start)
    print(f"acc {acc},")
    print(f"per {per},")
