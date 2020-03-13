from data_loader import Corpus
from encoding import Encoder
from gensim.models import KeyedVectors
import argparse

parser = argparse.ArgumentParser(
    description="In the future, train a LSTM language model using word embeddings.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--corpus-path",
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




if __name__ == "__main__":
    args = parser.parse_args()

    # Load the pre-trained embeddings
    embeddings = KeyedVectors.load_word2vec_format(args.vectors_path,
                                                    binary=True)
    # Load the corpora, find the vocabulary and what is in the embeddings.
    corpora = Corpus(args.corpus_path, embeddings)
    # Don't need the embeddings any longer. corpora has a copy of the relevant
    # vectors.
    del embeddings

    encoder = Encoder(50, len(corpora.vocab), corpora.vectors)

    model = RNNModel(encoder.encoding_size,350,len(corpora.vocab),2,encoder)
