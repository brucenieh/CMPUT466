import raw.readingfiles as readfiles
from models.ngrams import Ngrams
from util import evaluate, k_fold

if __name__ == '__main__':
    # readfiles.make_datasets()
    data_set = k_fold(10, 'develop.csv')

    ngrams_model = Ngrams(2)
    x = [['1','2','3','4','5'], ['4','6','4','6','2','400']]
    # ngrams_model.train(data_set[0].training_set)
    # ngrams_model.train(x)
    print(evaluate(ngrams_model, data_set[0].training_set, data_set[0].validation_set))

    # tweak hyper parameter n of n-gram model
    best_n = (1,0) #n value, accuracy
    for n in range(2,8):
        ngrams_model = Ngrams(n)
        accuracy = evaluate(ngrams_model, data_set[0].training_set, data_set[0].validation_set)
        print(n,accuracy)
        if accuracy > best_n[1]:
            best_n = (n,accuracy)
    
    print(best_n)
