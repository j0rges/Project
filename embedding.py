from gensim import KeyedVectors
from numpy import ndarray, zeros, hstack, matmul
from torch import Tensor, tensor, double


class Embedding(KeyedVectors):
    """ An embedding for neural language models, which uses pre-trained vector
        representations. Such as those obtained using the skip-gram model. This
        class is simpy the KeyedVectors class from the gensim library, with some
        added functionality.

        You can download pretrained embeddings from word2vec code archive.
    """

    def filter_vocab(self, new_vocab, add_tokens=True, unknown='mean'):
        """ Reduce the vocabulary of the embedding to a subset of it. You may
            want to do this for efficiency (much faster do do operations such as
            most_similar if the vocabulary is smaller).

            Parameters:

                new_vocab: is a list with all the words wanted in the
                    vocabulary.

                add_tokens: If True, tokens that aren't in the original
                    vocabulary, will be added and the dimensionality of the
                    embedding increased. Default: True.

                unknown: what to use as the main vector for tokens that are
                    added to the vocabulary. 'mean' is the only supported
                    option, and it is the mean of all other vectors.

            Returns: the new vector size (whether it has changed or not).
        """
        # new_vocab is a list with the words we want in our vocabulary
        representations = []
        new_tokens = []
        unknown_vector = self.vectors.mean(axis=0)
        for word in new_vocab:
            if word in self.vocab:
                vec = self.vectors[self.vocab[word].index]
                representations.append(vec)
            elif add_tokens:
                new_tokens.append(word)
                representations.append(unknown_vector)
            else:
                raise ValueError('The word "{}" is not in the initial voca'
                    'bulary and add_tokens=False.'.format(word))
        self.vectors = zeros((0,self.vector_size))
        self.index2entity = []
        self.vocab = {}
        self.add(new_vocab,representations)
        # Add the extra dimensions to all the vectors:
        if add_tokens and new_tokens:
            self.add_extra_dimensions(new_tokens)
            print("The words added to the vocabulary are: {}".format(new_tokens))
        return self.vector_size

    def add_extra_dimensions(self, new_tokens):
        """ When words are added to the vocabulary, increase the dimensionality
            of the embeddings to have an indicator column for each of the new
            words.
        """
        extra_dims = zeros(len(self.vocab), len(new_tokens))
        # Each new word has it's own indicator dimension. Similar to a one-hot
        # vector appended to the original vector representations.
        for i in range(len(new_tokens)):
            ind = self.vocab[new_tokens[i]].index
            extra_dims[ind, i] = 1
        self.vectors = hstack((self.vectors,extra_dims))
        self.vector_size = self.vector_size + len(new_tokens)

    def encode_word(self, word):
        """ Return the embedding for a given word.

            Input (word): either a string(the word) or an integer(the index for
            the word).

            Output: vector representation of the word as a numpy array.

            Errors: ValueError if the word isn't in the vocabulary or the index
                    is out of range.
        """
        valid_types = [int,str]
        word_type = type(word)

        if word_type == int:
            index = word
        elif word_type == str:
            if word in self.vocab:
                index = self.vocab[word].index
            else:
                raise ValueError("The word '{}' is not in the vocabulary".format(word))
        else:
            raise ValueError("The input had type '{}', which is not amongst the"
                  " supported types: {}".format(word_type,valid_types))
        return self.vectors[index]

    def encoder(self, inputs):
        """ Encode a batch of words.

            inputs (size (N,)) must be an itearable of words (either it's index
            or the string itself).

            returns a tensor with shape (N, vector_size), with the encoding of
            the words.
        """
        return tensor([self.encode_word(w) for w in inputs])

    def decoder(self, logits, tau):
        """ Turn a batch of outputs into a distribution over the vocabulary.
            Use the transpose of the encoding (inspired by Inan et al 2016):

            softmax((V^T x)/tau)

            where x is each row in logits.

            Parameters:

                logits: logits from a batch. Array like, with shape
                (batch_size, vector_size).

                tau: temperature parameter.

            Output: distribution of probability over the vocabulary for each
            element in the batch. torch tensor of type double.

        """
        logits = tensor(matmul(logits,self.vectors.transpose())/tau,
                        dtype=double)
        return logits.softmax(1)
