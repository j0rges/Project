import os, torch
from io import open

class Corpus(object):
    """ Object to tokenize and store the text corpus """

    def __init__(self, path, embedding):
        self.embedding = embedding
        self.narrow_vocab(os.path.join(path, 'train.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def narrow_vocab(self, path):
        """ Find the vocabulary of the train dataset and make the vocabulary of
            the embedding the same.
        """
        assert os.path.exists(path)

        vocab = []
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                [vocab.append(word) for word in words if not word in vocab]

        self.embedding.filter_vocab(vocab)


    def tokenize(self, path):
        """ Returns the word indices for a text file (dataset) """
        assert os.path.exists(path)

        word2idx = lambda x: self.embedding.vocab[x].index

        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(word2idx(word))
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def get_batch(data, i, batch_size):
    inputs = data[i*batch_size : (i+1)*batch_size]
    targets = data[i*batch_size + 1 : (i+1)*batch_size + 1]
    return inputs, targets
