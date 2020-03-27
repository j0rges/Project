import torch

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
        self.vectors = torch.tensor(vectors)
        self.default = torch.zeros(self.vectors.shape[1])
        self.input_size = vocab_size - len(vectors)
        self.hidden_size = size
        assert self.input_size > 0

        self.encoding_size = self.hidden_size + self.vectors.shape[1]
        self.linear = torch.nn.Linear(self.input_size, self.hidden_size)
        self.init_layer(self.linear)

    def encode1(self, inputs):
        if len(inputs.shape) == 0:
            if inputs >= len(self.vectors):
                return self.default
            else:
                return self.vectors[inputs]
        else:
            return torch.stack([self.encode1(val) for val in inputs])

    def encode2(self, inputs):
        if len(inputs.shape) == 0:
            x = torch.zeros(self.input_size)
            if inputs >= len(self.vectors):
                x[inputs - len(self.vectors)] = 1
            return x
        else:
            return torch.stack([self.encode2(val) for val in inputs])

    def forward(self, batch):
         x1 = self.encode1(batch).to(self.device)
         x2 = self.encode2(batch).to(self.device)
         x2 = self.linear(x2)
         return torch.cat((x1,x2),-1)

    def init_layer(self, layer):
      if hasattr(layer, "bias"):
        if type(layer.bias) != type(None):
            torch.nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            torch.nn.init.kaiming_normal_(layer.weight)

    @property
    def device(self):
        return next(self.parameters()).device
