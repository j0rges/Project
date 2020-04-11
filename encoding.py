import torch
F = torch.nn.functional

class Encoder(torch.nn.Module):
    """ Word encoder. If the word already has a representation, returns that
        plus the output of the linear with input 0-tensor. If the word
        doesn't have a representation, returns default with the output of
        the linear with input a one-hot vector for that word.
    """

    def __init__(self, size, vocab_size, vectors, default='zero'):
        """ size: number of units in the linear layer.

            vocab_size: number of words in the vocabulary.

            vectors: transfer learning vector representation for the first words
            in the vocabulary, in order.

            default: representation for the vectors part when the word doesn't
            have a pre-trained vector representation.
        """
        super().__init__()
        vectors = torch.tensor(vectors)
        self.input_size = vocab_size - len(vectors)
        assert self.input_size > 0
        self.hidden_size = size
        self.encoding_size = self.hidden_size + vectors.shape[1]
        self.linear = torch.nn.Linear(self.input_size, self.hidden_size)
        self.init_layer(self.linear)

        defaults = torch.zeros((self.input_size, vectors.shape[1]))
        self.vectors = torch.cat((vectors, defaults), 0)

        defaults = torch.zeros((vocab_size - self.input_size, self.input_size))
        one_hot = torch.zeros((self.input_size, self.input_size))
        # add the ones
        one_hot = one_hot.scatter(1,
                  torch.tensor([[i] for i in range(self.input_size)]), 1)
        self.linear_inputs = torch.cat((defaults, one_hot), 0)



    def encode1(self, inputs):
        return F.embedding(inputs, self.vectors)

    def encode2(self, inputs):
        return F.embedding(inputs, self.linear_inputs)

    def forward(self, batch):
         x1 = self.encode1(batch)
         x2 = self.encode2(batch)
         x2 = self.linear(x2)
         return torch.cat((x1,x2),-1)

    def init_layer(self, layer):
      if hasattr(layer, "bias"):
        if type(layer.bias) != type(None):
            torch.nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            torch.nn.init.kaiming_normal_(layer.weight)
