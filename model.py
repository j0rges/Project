# This code is based on the example word language model from the pyTorch github
# repository: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """ LSTM language model which uses pre-trained word vector representations
        as encoding.

        Args:
            encoding_size: The dimensions of the word representations.
            hidden_size: The number of features in the hidden state
            num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
                would mean stacking two LSTMs/GRUs together to form a `stacked RNN`,
                with the second RNN taking in outputs of the first RNN and
                computing the final results. Default: 1
            decoupled: whether there is a linear layer between the top most RNN
                layer and the output. Default: True
    """

    def __init__(self, encoding_size, hidden_size, output_size, num_layers,
                 encoder, rnn_type='LSTM', dropout=0, decoupled=True):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Embedding(ntoken, encoding_size)
        self.encoder = encoder
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(encoding_size, hidden_size, num_layers, dropout=dropout)
        else:
            raise ValueError( "An invalid option for rnn_type was supplied, "
                              "options are ['LSTM', 'GRU']")
        # if decoupled:
        #     self.decoder = nn.Linear(hidden_size, encoding_size)
        # else:
        #     self.decoder = lambda x : x
        #     if hidden_size != encoding_size:
        #         raise ValueError("When flagging decoupled as False, the "
        #         "encoding_size and the hidden_size must be the same.")
        self.decoder = nn.Linear(hidden_size, output_size)
        self.init_layer(self.decoder)
        self.rnn_type = rnn_type
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_layer(self, layer):
      if hasattr(layer, "bias"):
        if type(layer.bias) != type(None):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, input, hidden):
        # emb = self.drop(self.encoder(input)) -> Decide on dropout
        emb = self.encoder(input)
        emb = self.drop(emb)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        # See what we put here (decoder layer or not?)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                    weight.new_zeros(self.num_layers, bsz, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.hidden_size)
