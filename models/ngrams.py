from nltk.util import ngrams, pad_sequence
from nltk.probability import FreqDist
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

class Ngrams:
    def __init__(self, n):
        self.n = n

    def train(self, data):
        n = self.n
        # ngrams_list = []
        # for line in data:
        #     line = list(pad_both_ends(line, n=n))
        #     ngrams_list.extend(list(ngrams(line, n)))
        # print(ngrams_list)
        lm = MLE(n)
        train, vocab = padded_everygram_pipeline(n, data)
        lm.fit(train, vocab)
        print(lm.generate(text_seed=['4']))


    def predict():
        return 1