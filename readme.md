## About: ##

This is just a repository for the code I will use in the final year project of
my Meng Mathematics and Computer Science. It is related to language models,
using LSTMs and seeing how using already trained word embeddings affect the
language model.

## Reference material: ##

For more information on word embeddings, see:

  *  The [Word2Vec] code archive is a distribution of an implementation for
computing vector representations of words. It also has links to pre-trained
vector representations.

  * [Efficient Estimation of Word Representations in Vector Space] is the most
relevant paper for the subject.


For more information on language models, see:

  * [Class-based n-gram models of natural language] provides an introduction as
well as detailed explanation of n-gram models.

  * [LSTM Neural Networks for Language Modeling]. This is the paper that
introduced the LSTM recurrent neural network to language models. This is what I
will be working on.

I will be using [pyTorch] to implement and train my models, and use as reference
the code publicly distributed by Gulordava et al. ([link]: https://github.com/facebookresearch/colorlessgreenRNNs). To work with the word2vec vectors, I will
be using [gensim].

Also relevant to this project are the following papers:

  * [Using the Output Embedding to Improve Language Models]

  * [The emergence of number and syntax units in LSTM language models]



[Word2Vec]: https://code.google.com/archive/p/word2vec/

[Efficient Estimation of Word Representations in Vector Space]: http://arxiv.org/pdf/1301.3781.pdf

[Class-based n-gram models of natural language]: https://dl.acm.org/doi/pdf/10.5555/176313.176316?download=false

[LSTM Neural Networks for Language Modeling]: https://www.isca-speech.org/archive/interspeech_2012/i12_0194.html

[pyTorch]: https://pytorch.org

[gensim]: https://radimrehurek.com/gensim/

[Using the Output Embedding to Improve Language Models]: https://arxiv.org/abs/1608.05859

[The emergence of number and syntax units in LSTM language models]: https://arxiv.org/abs/1903.07435
