from gensim import KeyedVectors



class Embedding(KeyedVectors):
    
    def filter_vocab(self, new_vocab)
        # new_vocab is a list with the words we want in our vocabulary
        representations = []
        for word in new_vocab:
            if word in self.vocab:
                vec = self.vectors[self.vocab[entity].index]
                representations.append(vec)
            else:
                raise ValueError('the word {} is not in the initial vocabulary!'.format(word))
        self.index2entity = []
        self.index2word = self.index2entity
        self.vocab = {}
        self.add(new_vocab,weights)
