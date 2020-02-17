from gensim import KeyedVectors
from numpy import ndarray, zeros
from torch import Tensor, tensor


class Embedding(KeyedVectors):
    """ An embedding for neural language models, which uses pre-trained vector
        representations. Such as those obtained using the skip-gram model. This
        class is simpy the KeyedVectors class from the gensim library, with some
        added functionality.

        You can download pretrained embeddings from word2vec code archive.
    """

    def filter_vocab(self, new_vocab):
        """ Reduce the vocabulary of the embedding to a subset of it. You may
            want to do this for efficiency (much faster do do operations such as
            most_similar if the vocabulary is smaller).

            Input: new_vocab is a list with all the words wanted in the
                   vocabulary.
        """
        # new_vocab is a list with the words we want in our vocabulary
        representations = []
        for word in new_vocab:
            if word in self.vocab:
                vec = self.vectors[self.vocab[word].index]
                representations.append(vec)
            else:
                raise ValueError('the word "{}" is not in the initial vocabulary!'.format(word))
        self.index2entity = []
        self.vectors = zeros((0,self.vector_size))
        self.vocab = {}
        self.add(new_vocab,representations)


    def encoder(self, word):
        """ Return the embedding for a given word.

            Input (word): either a string or an integer or a one-hot vector.

            Output: vector representation of the word as a numpy array.

            Errors: ValueError if the word isn't in the vocabulary or the index
                    is out of range.
        """
        valid_types = [int,str,ndarray,list,Tensor]
        word_type = type(word)
        
        if word_type == int:
            index = word
        elif word_type == str:
            if word in self.vocab:
                index = self.vocab[word].index
            else:
                raise ValueError("The word '{}' is not in the vocabulary".format(word))
        # Finally check if it is a list or numpy/torch equivalent, ie a one-hot
        # vector.
        elif word_type in valid_types:
            raise NotImplementedError("Please provide the input as a string or "
                  " integer, one-hot vectors are not supported yet.")
        else:
            raise ValueError("The input had type '{}', which is not amongst the"
                  " supported types: {}".format(word_type,valid_types))
        return self.vectors[index]

    def decoder(self, vector):
        """ Turn the output into a distribution over the vocabulary. Perhaps use
            the transpose of the encoding (inspired by Press & Wolf 2016).

            Input (word): vector representation (output).

            Output: distribution of probability over the vocabulary.

            Errors: ?Error if the dimensions don't match.
        """
        raise NotImplementedError()
