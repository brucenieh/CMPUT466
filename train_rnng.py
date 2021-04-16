import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from util import k_fold,read_data
import random

from models.rnng import RNNG



def data_reader(data, batch_size):
    """Returns a generator to generate data batches from the whole dataset
    based on batch_size.

    Args:
        data (list of str): our whole dataset of sentences
        batch_size (int): size of sentences in each data batch

    Yields:
        list of str: data batch
    """
    for i in range(0, len(data), batch_size):
        
        yield data[i:i + batch_size]

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(mapping,model,data,batch_size,lr,epochs):
    """training function for RNNG model

    Args:
        mapping (dict): dictionary that maps word to its corresponding index
        model (nn.Module): Model to be trained
        data (list): a list of corpus in forms of a list of tokens
        batch_size (int): number of corpus to be read in one batch
        lr (float): learning rate of
        epochs: number of epochs to train
    """
    vocab = len(mapping)
    sentence_length = 50
    model.train()
    
    hidden = model.init_hidden(batch_size)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.NLLLoss()
    for _ in range(epochs):
        dr = data_reader(data, batch_size)
        for batch_data in dr:
            optimizer.zero_grad()
            print(len(batch_data))

            X_train = []
            Y_train = []
            for line in batch_data:
                if len(line) < sentence_length + 1:
                    continue
                vector = np.zeros(vocab)
                y_word = line[sentence_length]
                vector[mapping[y_word]] = 1

                sentence = line[:sentence_length+1]
                sentence = np.array([mapping[word] for word in sentence])
                
                X_train.append(sentence[0:-1])
                Y_train.append(vector)

            X_train = np.asarray(X_train)
            Y_train = np.asarray(Y_train)
            Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)

            hidden = repackage_hidden(model.init_hidden(len(X_train)))
            output, hidden = model(X_train,hidden)
            
            loss = criterion(output, Y_train)
            print('Loss:', loss)

            loss.backward()
            optimizer.step()

def predict(model, data, mapping):
    """predicts the next word using our ANN

    Args:
        model (nn.Module): language model to be used for predicting
        data (list): a list of corpus in forms of a list of tokens
        mapping (dict): dictionary that maps word to its corresponding index

    Returns:
        np.ndarray: an array of prediction's probability distribution with size
                    [batch_size x vocab size]
    """
    sentence_length = 50
    X_test = []
    # iterate each review
    for line in data:
        sentence = []
        # iterate each word in a review
        for i in range(sentence_length):
            # try looking up the word in vocab mapping
            try:
                sentence.append(mapping[line[i]])
            # if word does not exist in mapping
            except KeyError:
                # if first word is unknown, assign random index, else copy
                # index of previous word
                if len(sentence) == 0:
                    sentence.append(random.randint(0, model.vocab_size - 1))
                else:    
                    sentence.append(sentence[-1])
        X_test.append(sentence)
    # convert list to Numpy array
    X_test = np.asarray(X_test)

    model.eval()
    hidden = model.init_hidden(len(X_test))
    with torch.no_grad():
        output, hidden = model(X_test, hidden)
        hidden = repackage_hidden(hidden)
    
    return output

def evaluate(model,testing_data,mapping):
    """Returns the accuracy and perplexity of the RNNG model
    Args:
        model (Model): ANN Model
        testing_data (list of str): Data for the model to be tested on
        mapping (dictionary): Vocabulary of model
    Returns:
        float: Accuracy of the model
        float: Perplexity of the model
    """
    accuracy = 0
    perplexity = 0
    sentence_length = 50
    filtered_data = []
    targets = []
    for test in testing_data:
        # skip sentences shorter than 51 words
        if len(test) < sentence_length + 1:
            continue
        filtered_data.append(test[:sentence_length])
        targets.append(test[sentence_length])
    # predict using filtered data
    filtered_data = np.asarray(filtered_data)
    prob_distrib = predict(model,filtered_data,mapping)
    
    for i in range(len(prob_distrib)):
        target = targets[i]
        distrib = prob_distrib[i][-1]
        distrib = distrib/np.linalg.norm(distrib, ord=1)
        
        # check if word is correct
        try:
            correct_index = mapping[target]
        except:
            perplexity += 1/0.000001

        # compare prediction from model and update accuracy
        if np.argmax(distrib) == correct_index:
            accuracy += 1
        # update perplexity
        try:
            perplexity += 1/distrib[correct_index]
        except:
            perplexity += 1/0.000001

    accuracy = accuracy/len(filtered_data)
    perplexity = perplexity/len(filtered_data)

    return accuracy,-float(perplexity)

def main():
    data_set = k_fold(10, 'develop.csv')
    weights_matrix = pickle.load(open('vocab_embedding.pkl', 'rb'))
    with open('vocab.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    batch_size = 200
    model = RNNG(weights_matrix,100,1,len(mapping),batch_size,0.2)
    train(mapping,model,data_set[0].training_set,batch_size,0.1,1)
    torch.save(model, 'RNNG.pt')
    
    # For model evaluation
    model = torch.load('RNNG.pt')
    acc,perplex = evaluate(model,data_set[0].validation_set,mapping)
    print(f'Accuracy: {acc}\nPerplexity: {perplex}')


if __name__ == '__main__':
    main()