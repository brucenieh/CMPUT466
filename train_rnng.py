import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from util import k_fold,read_data
import random

from models.rnng import RNNG



def data_reader(data, batch_size,sentence_length):
    for i in range(0, len(data), batch_size):
        
        yield data[i:i + batch_size]

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(mapping,model,data,batch_size,lr):
    vocab = len(mapping)
    sentence_length = 50
    model.train()
    dr = data_reader(data, batch_size,sentence_length)
    hidden = model.init_hidden(batch_size)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.NLLLoss()
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
            # vector = mapping[y_word]
            sentence = line[:sentence_length+1]
            sentence = np.array([mapping[word] for word in sentence])
            
            X_train.append(sentence[0:-1])
            # y_one_hot = np.zeros([50,vocab])
            # start_index = 0
            # for index in sentence[1:]:
            #     y_one_hot[start_index][index] = 1
            #     start_index += 1
            # result = np.sum(y_one_hot,axis=1).sum()
            # print(result)
            Y_train.append(vector)
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        print(X_train.shape)
        print(Y_train.shape)
        Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)

        hidden = repackage_hidden(model.init_hidden(len(X_train)))
        output, hidden = model(X_train,hidden)
        print('output',output.shape)
        print('y train',Y_train.shape)

        loss = criterion(output, Y_train)
        print(loss)
        loss.backward()
        optimizer.step()

def predict(model, data, mapping):
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
    
    print(len(prob_distrib))

    for i in range(len(prob_distrib)):
        target = targets[i]
        distrib = prob_distrib[i][-1]
        distrib = distrib/np.linalg.norm(distrib, ord=1)
        # print(np.sum(distrib))
        print(distrib.shape)

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

    return accuracy,perplexity

def main():
    data_set = k_fold(10, 'develop.csv')
    weights_matrix = pickle.load(open('vocab_embedding.pkl', 'rb'))
    with open('vocab.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    batch_size = 200
    model = RNNG(weights_matrix,100,1,len(mapping),batch_size,0.2)
    train(mapping,model,data_set[0].training_set[:100],batch_size,0.1)
    torch.save(model, 'RNNG.pt')
    
    # For model evaluation
    model = torch.load('RNNG.pt')
    result = evaluate(model,data_set[0].validation_set,mapping)
    print(result)


if __name__ == '__main__':
    main()