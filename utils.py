import pickle, os, time, math
from datetime import datetime
from train_functions import evaluate
from data_loader import Corpus
from encoding import Encoder
from model import RNNModel
from old_model import RNNModel as old_model

def save_checkpoint(model, path, valid_loss, args={}):
    if path:
        to_save = {'params' : model.state_dict(), 'valid_loss': valid_loss,
                   'args': args}
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)
        print('checkpoint saved to {}'.format(path))

def load_checkpoint(path):
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

class Logger(object):

    def __init__(self, path):
        self.create_files(path)

    def create_files(self, path):
        if not os.path.exists(path):
            raise RuntimeError("the folder {} doesn't exist.")
        self.base_path = path
        self.train_log_file = os.path.join(path, 'train.csv')
        self.valid_log_file = os.path.join(path,'valid.csv')
        with open(self.train_log_file, 'w') as f:
            f.write("epoch,batches,time,loss,perplexity\n")
            f.write("nan,nan,{},nan,nan\n".format(time.time()))
        with open(self.valid_log_file,'w') as f:
            f.write("epoch,time,loss,perplexity\n")
            f.write("nan,{},nan,nan\n".format(time.time()))

    def log_valid(self, epoch, loss):
        line = '{},{},{},{}\n'.format(epoch, time.time(), loss, math.exp(loss))
        with open(self.valid_log_file, 'a') as f:
            f.write(line)

    def log_train(self, epoch, batches, loss):
        line = '{},{},{},{},{}\n'.format(epoch, batches, time.time(),
                                         loss, math.exp(loss))
        with open(self.train_log_file, 'a') as f:
            f.write(line)

    def log_description(self, args):
        path = os.path.join(self.base_path, 'description.txt')
        args = args.__dict__
        with open(path, 'w') as fout:
            fout.write(str(datetime.now()) + '\n\n')
            for key, val in args.items():
                fout.write('{}: {}\n'.format(key,val))


def load_model_corpora(checkpoint):
    """ Load the model the checkpoint pointed at by `checkpoint' is for and the
        corpora indicated in the arguments within the checkpoint.
    """
    try:
        checkpoint = load_checkpoint(checkpoint)
        args = checkpoint['args']
        params = checkpoint['params']
    except Exception as e:
        print('The following exception ocurred:')
        print(e)
        raise RuntimeError('The first object in checkpoint must be a '
              'dictionary containing at least [args,params].')
    # Use the arguments to create a model that is the same as the one we have
    # the parameters for.
    if args.load:
        with open(args.load,'rb') as f:
            stored_dict = pickle.load(f)
        corpora = Corpus(args.corpus,load=True,vocab=stored_dict['vocabulary'],
                   vectors=stored_dict['vectors'])
    else:
        # I never do load = False.
        corpora = None
    if not hasattr(args, 'old_model'):
        args.old_model = False
    if args.old_model:
        model = old_model('LSTM', len(corpora.vocab), args.encoder_size,
                    args.hidden_size, args.layers, args.dropout)
    else:
        encoder = Encoder(50, len(corpora.vocab), corpora.vectors)
        model = RNNModel(encoder.encoding_size, args.hidden_size,
                    len(corpora.vocab), args.layers, encoder, dropout=args.dropout)
    # load the parameters from checkpoint
    model.load_state_dict(params)
    return model, corpora
