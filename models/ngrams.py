from nltk.util import ngrams, pad_sequence
from nltk.probability import FreqDist
from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import pad_both_ends
from collections import Counter, defaultdict

class Ngrams:
    def __init__(self, n):
        """Our n-gram model uses a default dict of default dict as our data
        structure. The outer default dict consists of n-1 words tuples as keys
        and default dicts as values. The inner default dict consists of n-th
        words as keys and probabilities as values. For example:
        self.model = {
            ('This', 'is'): { 'a': 1.0 },
            ('is', 'a'): { 'good': 0.5, 'bad': 0.5 }
        }

        Args:
            n (int): n in n-gram

        Raises:
            Exception: Raised when n is not 2 to 20
        """
        if n not in range(2,21):
            raise Exception("Invalid n, please use 2-20")
        self.n = n
        self.model = defaultdict(lambda: defaultdict(lambda: 1))

    def train(self, data):
        """
        Trains our n-gram model by padding the start and end of the sentences,
        getting n-grams as tuples using NLTK's built-in functions. Stores the
        count in our model structure and calculates probability distribution of
        the n-th word.

        Args:
            data (list of str): our training data
        """
        # pad, extract n-grams from training data and store it in ngrams_list
        ngrams_list = []
        for line in data:
            line = list(pad_both_ends(line, n=self.n))
            ngrams_list.extend(list(ngrams(line, self.n)))

        # store counts in self.model[n-1 words tuple][n-th word]
        if self.n == 2:
            for ngram in ngrams_list:
                self.model[ngram[0]][ngram[-1]] += 1
        else:
            for ngram in ngrams_list:
                self.model[ngram[:-1]][ngram[-1]] += 1
        # add a tag for unknown words with count one, then calculate probability
        # distribution from count by iterating through each n-1 grams tuples and
        # divide by the total number of n-th word
        for x in self.model:
            self.model[x]['<UNK>'] = 1
            total_count = float(sum(self.model[x].values()))
            for y in self.model[x]:
                self.model[x][y] /= total_count

    def predict(self, text, num_words=1):
        """Predicts the next word based on the most probable words in our model,
        returns the top num_words words found and probability distribution of
        the n-th word with add one smoothing in a defaultdict with key: value
        'word': 'probability' 

        Args:
            text (str): sentence from testing data to predict next word for
            num_words (int, optional): Specifies the number of most probable
            words to return. Defaults to 1.

        Returns:
            list of str: list of most probable words from our model
            defaultdict: probability distribution of the n-th word
        """
        # only selects the last n-1 words to search in our model
        if len(text) > self.n:
            text = text[-self.n+1:]
        if self.n == 2:
            text = text[0]
        else:
            text = tuple(text)

        # sort next words from most probable to least probable
        word_count = sorted(self.model[text].items(), key=lambda item: item[1],
                            reverse=True)
        top_word_count = word_count[:num_words]
        words = [word[0] for word in top_word_count]
        # if next word not found, return . as fallback
        if len(words) == 0:
            words = ['.']
        prob_dist = self.model[text]
        return words, prob_dist