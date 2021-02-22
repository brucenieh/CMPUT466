import raw.readingfiles as readfiles
from models.ngrams import Ngrams

if __name__ == '__main__':
    # readfiles.make_datasets()
    ngrams_model = Ngrams(2)
    x = [['1','2','3','4','5'], ['4','6','4','6']]
    ngrams_model.train(x)
