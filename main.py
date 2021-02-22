import raw.readingfiles as readfiles
from models.ngrams import Ngrams
from k_fold import k_fold

if __name__ == '__main__':
    # readfiles.make_datasets()
    data_set = k_fold(10, 'develop.csv')

    ngrams_model = Ngrams(2)
    x = [['1','2','3','4','5'], ['4','6','4','6','2','400']]
    ngrams_model.train(data_set[0].training_set)
    # ngrams_model.train(x)
    print(ngrams_model.predict('a'))
