import pickle

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
