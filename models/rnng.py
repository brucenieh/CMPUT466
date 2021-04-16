import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def create_emb_layer(weights_matrix, non_trainable=False):
    """creates embedding layer covering the union of glove embedding and training vocab

    Args:
        weights_matrix (dict): vocab embedding as a dictionary loaded from vocab_embedding.pkl
        non_trainable (bool, optional): Whether allow the embedding weights to be trainable. Defaults to False.

    Returns:
        nn.embedding, int, int: a pytorch embedding layer, number of embeddings and embedding dimensions
                                we are currently using 50 embeddings for each word
    """
    weights_matrix = torch.from_numpy(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class RNNG(nn.Module):
    # This model is implemented with reference to:
    # https://github.com/kmkurn/pytorch-rnng
    def __init__(self, weights_matrix, hidden_size, num_layers,vocab_size,batch_size,dropout,num_actions=3):
        super(RNNG,self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_actions = num_actions
        action_embedding_size = 30

        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.action_embedding = nn.Embedding(self.num_actions, action_embedding_size)
        self.action2encoder = nn.Sequential(nn.Linear(action_embedding_size, self.hidden_size),nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_size,vocab_size)

        
    def forward(self, inp, hidden):
        inp = torch.from_numpy(inp)
        inp = torch.tensor(inp).to(torch.int64)
        out = self.embedding(inp)
        out,hidden = self.gru(out, hidden)
        out = self.drop(out)
        out = self.lin(out)
        out = F.log_softmax(out, dim=1)
        
        return out,hidden
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Pytorch RNN model for reference
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)




