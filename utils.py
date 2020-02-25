import torch


def save_checkpoint(model, path, other={}, verbose=True):
    """ Save a checkpoint of the parameters in the file pointed to by path,
        and whatever extra stuff passed through other (must be a dictionary).
    """
    torch.save({'parameters': model.model.state_dict()}.update(other),
                path)
    if verbose:
        print('Parameters saved to {}'.format(path))

def load_checkpoint(model, path, kw='parameters', device=torch.device("cpu")):
    checkpoint = torch.load(path, map_location = device)
    model.load_state_dict(checkpoint[kw])
