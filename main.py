import raw.readingfiles as readfiles
import pickle, time
from models.ngrams import Ngrams
from models.ANN import ANN
from models.rnn import RNN
from models.rnng import RNNG
from train_rnng import train, evaluate
from util import initial_setup, read_data, evaluate_ngrams, \
                 evaluate_ANN, evaluate_RNN


if __name__ == '__main__':
    initial_setup() # This line needs to be executed only once
    training_set = read_data('training_data.csv')
    testing_set = read_data('testing_data.csv')

    ngrams_model = Ngrams(2)
    accuracy,perplexity = evaluate_ngrams(ngrams_model, training_set, testing_set, 50)
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

    # Load glove embedding
    weights_matrix = pickle.load(open('vocab_embedding.pkl', 'rb'))
    with open('vocab.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    RNNG_model = RNNG(weights_matrix,hidden_size=100,num_layers=2,
                      vocab_size=len(mapping),batch_size=200,dropout=0.2)
    train(mapping,RNNG_model,training_set,200,0.1,1)
    acc, per = evaluate(RNNG_model,testing_set,mapping)
    print("RNNG accuracy, perplexity: ", acc, per)
