import raw.readingfiles as readfiles
from models.ngrams import Ngrams
from models.ANN import ANN
from util import build_vocab, evaluate, k_fold, read_data, evaluate_ANN

def data_reader(data,batch_size):
    # datas = []
    # print(len(data)//batch_size)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

if __name__ == '__main__':
    # readfiles.make_datasets()
    data_set = k_fold(10, 'training_data.csv')
    # training_set = read_data('training_data.csv')
    # testing_set = read_data('testing_data.csv')


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
    ANN_model = ANN()
    
    
    print(len(ANN_model.mapping))
    print("done vocab")

    
    counter = 1
    total = int(len(data_set[0].training_set)/1000) + 1
    for _ in range(ANN_model.epoch):
        dr = data_reader(data_set[0].training_set, 1000)
        for data in dr:
            print(f"Batch {counter} of {total}")
            counter += 1
            ANN_model.train(data)
        # print(data)
    ANN_model.save()
    
    # ANN_model.train(data_set[0].training_set[-1000:])
    # ANN_model.save()
    print(evaluate_ANN(ANN_model, data_set[0].training_set[-100:], ANN_model.mapping))
    # print(ANN_model.mapping['test'])
    # print(training_set[:5])
    # build_vocab(data_set[0].training_set, 'vocab')
    # ANN_model.train(data_set[0].training_set[:5000])
    # print(data_set[0].validation_set)
    # for i in range(10):
    #     ANN_model.predict(data_set[0].validation_set[i][:50])
    # ANN_model.train(["abc"])
