from nltk.util import ngrams, pad_sequence
from nltk.probability import FreqDist
from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import pad_both_ends
from collections import Counter, defaultdict

class Ngrams:
    def __init__(self, n):
        if n not in range(2,21):
            raise Exception("Invalid n, please use 2-20")
        self.n = n

    def train(self, data):
        """[summary]

        Args:
            data ([type]): [description]
        """
        n = self.n
        ngrams_list = []
        for line in data:
            line = list(pad_both_ends(line, n=n))
            ngrams_list.extend(list(ngrams(line, n)))

        # https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        if n == 2:
            for ngram in ngrams_list:
                self.model[ngram[0]][ngram[-1]] += 1
        else:
            for ngram in ngrams_list:
                self.model[ngram[:-1]][ngram[-1]] += 1
        
        # for x in model:
        #     total_count = float(sum(model[x].values()))
        #     for y in model[x]:
        #         model[x][y] /= total_count

        # print(model)


    def predict(self, text, num_words=1):
        if len(text) > self.n:
            text = text[-self.n+1:]
        if self.n == 2:
            text = text[0]
        else:
            text = tuple(text)
        word_count = sorted(self.model[text].items(), key=lambda item: item[1], reverse=True)
        top_word_count = word_count[:num_words]
        words = [word[0] for word in top_word_count]
        if len(words) == 0:
            return ['.']
        return words