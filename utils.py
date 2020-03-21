import pickle, os, time, math

def save_checkpoint(model, path, valid_loss, args={}):
    if path:
        to_save = {'params' : model.state_dict().cpu(), 'valid_loss': valid_loss,
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
        if os.path.exists(path):
            raise RuntimeError("the folder {} already exists. Are you sure you"
            " not overwritting some logs?")
        os.mkdir(path)
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
