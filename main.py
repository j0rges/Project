from data_loader import Corpus
from encoding import Encoder
from train_functions import Trainer, evaluate
from model import RNNModel
from old_model import RNNModel as old_model
from utils import save_checkpoint, Logger
from datetime import datetime
import argparse, math, pickle, torch

parser = argparse.ArgumentParser(
    description="In the future, train a LSTM language model using word embeddings.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument(
    "--corpus", default="wikitext-2/", type=str,
    help="Path where train.txt, valid.txt and test.txt are contained.")
parser.add_argument("--embedding-path", default="vectors.pkl", type=str,
                    help="Path for the binary file containing the embeddings.")
parser.add_argument("--epochs", default=20, type=int,
                    help='Number of epochs to train for.')
parser.add_argument("--lr", default=20, type=int, help='learning rate.')
parser.add_argument("--batch-size", default=64, type=int,
                    help='Number of batches to divide the data in.')
parser.add_argument("--seq-len", default=35, type=int,
                    help='length of the training sequences (backpropagation '
                    'will be truncated to this number of steps).')
parser.add_argument("--dropout", default=0.5, type=float,
                     help='dropout of the network.')
parser.add_argument("--clip-grad", default=0.25, type=float,
                     help='gradient clipping.')
parser.add_argument("--layers", default=1, type=int,
                    help='Number of stacked RNN layers.')
parser.add_argument("--hidden-size", default=350, type=int,
                    help='The number of units each RNN layer has.')
parser.add_argument("--load", default='', type=str,
                    help='If provided, the path with vocabulary and vectors.')
parser.add_argument("--checkpoint", default='', type=str,
                    help='Path to store checkpoints of the model during training.')
parser.add_argument("--log-dir", default='', type=str,
                    help='If provided, logs will be stored in the directory.')
parser.add_argument("--log-interval", default=100, type=int,
                    help='Number of batches between information is logged.')
parser.add_argument("--old-model", action='store_true')
parser.add_argument("--encoder-size", type=int, default=400)
parser.add_argument("--dataset-portion", type=float, default=1,
                    help="If provided, this is the proportion of the training "
                    "set to be used in training.")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.log_dir:
        logger = Logger(args.log_dir)
        logger.log_description(args)
    else:
        logger = None
    # if available use a GPU.
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.load:
        with open(args.load,'rb') as f:
            stored_dict = pickle.load(f)
        corpora = Corpus(args.corpus,load=True,vocab=stored_dict['vocabulary'],
                   vectors=stored_dict['vectors'], portion=args.dataset_portion)
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

    if args.old_model:
        model = old_model('LSTM', len(corpora.vocab), args.encoder_size,
                    args.hidden_size, args.layers, args.dropout)
    else:
        encoder = Encoder(50, len(corpora.vocab), corpora.vectors)
        model = RNNModel(encoder.encoding_size, args.hidden_size,
                    len(corpora.vocab), args.layers, encoder, dropout=args.dropout)

    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, corpora, criterion, device, logger,
              args.batch_size, args.seq_len, args.lr, args.log_interval,
              args.clip_grad)
    best_valid_loss = float("inf")
    for epoch in range(args.epochs):
        print('Time at the start of epoch {} is {}'.format(epoch,datetime.now()))
        trainer.train()
        valid_loss = evaluate(model,corpora, criterion, device)
        print('Validation loss: {:.2f}. Perplexity: {:.2f}'.format(valid_loss,
              math.exp(valid_loss)))
        if args.log_dir:
            logger.log_valid(epoch, valid_loss)
        save_checkpoint(model.to(torch.device('cpu')), args.checkpoint,
                        valid_loss, args)
        model = model.to(device)

        # Anneal the learning rate if the validation loss hasn't improved.
        if (valid_loss - best_valid_loss) < -0.01:
            best_valid_loss = valid_loss
        else:
            trainer.learning_rate /= 4.0
