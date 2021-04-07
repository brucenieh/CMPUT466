import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from util import evaluate, k_fold,read_data

from models.rnng import RNNG



def data_reader(data, batch_size,sentence_length):
    for i in range(0, len(data), batch_size):
        data = data[i:i + batch_size]
        
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
    for batch_i, batch_data in enumerate(dr):
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
            sentence = line[:sentence_length]
            sentence = np.array([mapping[word] for word in sentence])
            
            X_train.append(sentence)
            Y_train.append(vector)
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        Y_train = torch.from_numpy(Y_train)

        hidden = repackage_hidden(model.init_hidden(len(X_train)))
        output, hidden = model(X_train,hidden)
        print(output[0].shape)
        print(Y_train.shape)

        loss = criterion(output[0], Y_train.flatten())
        print(loss)
        loss.backward()
        optimizer.step()

def main():
    data_set = k_fold(10, 'develop.csv')
    weights_matrix = pickle.load(open('vocab_embedding.pkl', 'rb'))
    with open('vocab.pkl', 'rb') as f:
        mapping = pickle.load(f)
    batch_size = 10
    model = RNNG(weights_matrix,100,1,len(mapping),batch_size)
    train(mapping,model,data_set[0].training_set,batch_size,0.1)

if __name__ == '__main__':
    main()