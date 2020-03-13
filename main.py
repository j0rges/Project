from data_loader import Corpus
from encoding import Encoder
from gensim.models import KeyedVectors
import argparse

parser = argparse.ArgumentParser(
    description="In the future, train a LSTM language model using word embeddings.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--corpus",
    default="wikitext-2/",
    type=str,
    help="Path where train.txt, valid.txt and test.txt are contained."
)
parser.add_argument(
    "--embedding-path",
    default="vectors.pkl",
    type=str,
    help="Path for the binary file containing the embeddings."
)
parser.add_argument("--epochs", default=20, type=int,
                help='Number of epochs to train for.')
parser.add_argument("--lr", default=20, type=int, help='learning rate.')
parser.add_argument("--batch-size", default=64, type=int,
                    help='Number of batches to divide the data in.')
parser.add_argument("--seq-len", default=35, type=int,
                    help='length of the training sequences (backpropagation '
                    'will be truncated to this number of steps).')
parser.add_argument("--layers", default=1, type=int,
                    help='Number of stacked RNN layers.')
parser.add_argument("--hidden-size", default=350, type=int,
                    help='The number of units each RNN layer has.')
parser.add_argument("--load", default='', type=str,
                    help='If provided, the path with vocabulary and vectors.')




if __name__ == "__main__":
    args = parser.parse_args()

    if args.load:
        with open(load,'rb') as f:
            stored_dict = pickle.load(f)
        corpora = Corpus(args.corpus,load=True,vocab=stored_dict['vocabulary'],
                         vectors=stored_dict['vectors'])
    else:
        # Load the pre-trained embeddings
        embeddings = KeyedVectors.load_word2vec_format(args.vectors_path,
                                                        binary=True)
        # Load the corpora, find the vocabulary and what is in the embeddings.
        corpora = Corpus(args.corpus, embeddings)
        # Don't need the embeddings any longer. corpora has a copy of the relevant
        # vectors.
        del embeddings

    encoder = Encoder(50, len(corpora.vocab), corpora.vectors)

    model = RNNModel(encoder.encoding_size, args.hidden_size,
                    len(corpora.vocab), args.layers, encoder)
