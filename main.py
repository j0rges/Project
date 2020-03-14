from data_loader import Corpus
from encoding import Encoder
from train_functions import train, evaluate
from model import RNNModel
import argparse, math, pickle, torch

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
parser.add_argument("--checkpoint", default='', type=str,
                    help='Path to store checkpoints of the model during training.')
parser.add_argument("--log-interval", default=100, type=int,
                    help='Number of batches between information is logged.')


def save_checkpoint(model, path, valid_loss, more={}):
    if path:
        to_save = {'params' : model.state_dict(),
                   'valid_loss': valid_loss}.update(more)
        with open(path, 'wb') as f:
            pickle.dump(to_save,f)
        print('checkpoint saved to {}'.format(path))


if __name__ == "__main__":
    args = parser.parse_args()

    if args.load:
        with open(args.load,'rb') as f:
            stored_dict = pickle.load(f)
        corpora = Corpus(args.corpus,load=True,vocab=stored_dict['vocabulary'],
                         vectors=stored_dict['vectors'])
    else:
        # Load the pre-trained embeddings
        from gensim.models import KeyedVectors
        embeddings = KeyedVectors.load_word2vec_format(args.vectors_path,
                                                        binary=True)
        # Load the corpora, find the vocabulary and what is in the embeddings.
        corpora = Corpus(args.corpus, embeddings)
        # Don't need the embeddings any longer. corpora has a copy of the relevant
        # vectors.
        del embeddings

    criterion = torch.nn.CrossEntropyLoss()

    encoder = Encoder(50, len(corpora.vocab), corpora.vectors)

    model = RNNModel(encoder.encoding_size, args.hidden_size,
                    len(corpora.vocab), args.layers, encoder)

    best_valid_loss = float("inf")
    lr = args.lr
    for epoch in range(args.epochs):
        train(model, corpora, criterion, epoch, batch_size=args.batch_size,
              seq_len=args.seq_len, learning_rate=lr,
              log_interval=args.log_interval)
        valid_loss = evaluate(model,corpora, criterion)
        print('Validation loss: {:.2f}. Perplexity: {:.2f}'.format(valid_loss,
              math.exp(valid_loss)))
        save_checkpoint(model, args.checkpoint, valid_loss)

        # Anneal the learning rate if the validation loss hasn't improved.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        else:
            lr /= 4.0
