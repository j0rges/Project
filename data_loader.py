import os, torch
from io import open

class Corpus(object):
    """ Object to tokenize and store the text corpus

        path: path to the directory with the training, validation and test
            datasets ('train.txt','valid.txt','test.txt')
        embedding: gensim KeyedVectors object.
        vocab: dictionary from word to index (should contain all words in the
            vocabulary, with the ones with vector representations first).
        vectors: vector representations.
        load: whether to load the vocab and vectors, or derive from traing
            corpora. If true, vocab and vectors must be provided, otherwise,
            embedding should be provided.
    """

    def __init__(self, path, embedding=None, vocab=None, vectors=None, load=False):
        if load:
            self._load(vocab,vectors)
        else:
            self.narrow_vocab(os.path.join(path, 'train.txt'), embedding)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def _load(self, vocab, vectors):
        self.vocab = vocab
        self.vectors = vectors

    def narrow_vocab(self, path, embeddings):
        """ Find the vocabulary of the train dataset and make the vocabulary of
            the embedding the same.
        """
        assert os.path.exists(path)

        # Find all the distinct words in the file.
        vocabulary = set()
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                [vocabulary.add(word) for word in words if not word in vocabulary]

        # Find words in embedding and words not in it.
        emb_vocab = {}
        vocab_out = {}
        vectors = []
        for word in vocabulary:
            if word in embeddings.vocab:
                emb_vocab[word] = len(emb_vocab)
                vectors.append(embeddings.vectors[embeddings.vocab[word].index])
            else:
                vocab_out[word] = len(vocab_out)
        l = len(emb_vocab)
        # Create a single dictionary from word to its index.
        emb_vocab.update({key: value + l for key, value in vocab_out.items()})
        self.vocab = emb_vocab
        self.vectors = vectors

    def tokenize(self, path):
        """ Returns the word indices for a text file (dataset) """
        assert os.path.exists(path)

        word2idx = lambda x: self.vocab[x]

        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = [word2idx(word) for word in words]
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len=50):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
