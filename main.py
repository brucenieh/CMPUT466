import raw.readingfiles as readfiles
import pickle, time
from models.ngrams import Ngrams
from models.ANN import ANN
<<<<<<< HEAD
from util import build_vocab, evaluate, k_fold, read_data, evaluate_ANN
=======
from util import evaluate, k_fold,read_data, evaluate_ANN
>>>>>>> 1496a44f7b530051f05e5f16128feabcabc8f169

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

    # build_vocab(data_set[0].training_set, 'vocab')
    # ANN_model = ANN()    
    # print(len(ANN_model.mapping))
    # print("done vocab")

    
    # counter = 1
    # total = int(len(data_set[0].training_set)/1000) + 1
    # for _ in range(ANN_model.epoch):
    #     dr = data_reader(data_set[0].training_set, 1000)
    #     for data in dr:
    #         print(f"Batch {counter} of {total}")
    #         counter += 1
    #         ANN_model.train(data)
    #     # print(data)
    # ANN_model.save()
    
    # ANN_model.train(data_set[0].training_set)
    # print(evaluate_ANN(ANN_model, data_set[0].validation_set[:100], ANN_model.mapping))

    # performance = []
    # epochs = [5, 10, 15, 25, 50]
    # # epochs = [1, 2]
    # for epoch in epochs:
    #     start = time.time()
    #     ANN_model = ANN(epoch=epoch)
    #     ANN_model.train(data_set[0].training_set)
    #     accuracy, perplexity = evaluate_ANN(ANN_model,
    #                                     data_set[0].validation_set,
    #                                     ANN_model.mapping)
    #     performance.append([epoch,accuracy,perplexity])
    #     end = time.time()
    #     print(end - start)
    #     print([epoch,accuracy,perplexity])

    # with open('ANN_performance.pkl', 'wb') as f:
    #     pickle.dump(performance, f, pickle.HIGHEST_PROTOCOL)

    # with open('ANN_performance.pkl', 'rb') as f:
    #     test = pickle.load(f)
    #     print(test)

    # performance = []
    # lrs = [0.001, 0.01, 0.1, 0.5, 1, 2]
    # for lr in lrs:
    #     ANN_model = ANN(epoch=5, lr=lr)
    #     ANN_model.train(data_set[0].training_set)
    #     accuracy, perplexity = evaluate_ANN(ANN_model,
    #                                     data_set[0].validation_set,
    #                                     ANN_model.mapping)
    #     performance.append([lr,accuracy,perplexity])
    #     print([lr,accuracy,perplexity])

    # with open('ANN_performance_lr.pkl', 'wb') as f:
    #     pickle.dump(performance, f, pickle.HIGHEST_PROTOCOL)

    # with open('ANN_performance_lr.pkl', 'rb') as f:
    #     test = pickle.load(f)
    #     print(test)

    # performance = []
    # epochs = [5, 10, 15, 25, 50]
    # lrs = [0.001, 0.01, 0.1, 0.5, 1, 2]
    # batch_size = [2000, 1500, 1000, 750, 500]
    # for e in epochs:
    #     for lr in lrs:
    #         for bs in batch_size:
    #             ANN_model = ANN(epoch=e, lr=lr, batch_size=bs)
    #             ANN_model.train(data_set[0].training_set)
    #             accuracy, perplexity = evaluate_ANN(ANN_model,
    #                                             data_set[0].validation_set,
    #                                             ANN_model.mapping)
    #             performance.append([e,lr,bs,accuracy,perplexity])
    #             print([e,lr,bs,accuracy,perplexity])

    # with open('ANN_performance_all.pkl', 'wb') as f:
    #     pickle.dump(performance, f, pickle.HIGHEST_PROTOCOL)

    # with open('ANN_performance_all.pkl', 'rb') as f:
    #     test = pickle.load(f)

    start = time.time()
    ANN_model = ANN(epoch=5, lr=0.1, batch_size=2000)
    ANN_model.train(data_set[0].training_set)
    accuracy, perplexity = evaluate_ANN(ANN_model,
                                    data_set[0].validation_set,
                                    ANN_model.mapping)
    print("time", time.time() - start)
    print(accuracy, perplexity)


    # ANN_model.save()
    # print(ANN_model.mapping['test'])
    # print(training_set[:5])
    ANN_model.train(training_set)
    print(evaluate_ANN(ANN_model, testing_set[-100:], ANN_model.mapping))
    # ANN_model.train(["abc"])
