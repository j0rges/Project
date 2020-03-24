import pickle, os, time, math
from datetime import datetime

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
