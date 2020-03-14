import time, math, torch
from data_loader import batchify, get_batch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(model, corpus, criterion, epoch, device, batch_size = 25, seq_len = 35,
          learning_rate = 20, log_interval=100, clip_grad= 0.25):
    model = model.to(device)
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    number_tokens = len(corpus.vocab)
    train_data = batchify(corpus.train, batch_size, device)
    hidden = model.init_hidden(batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
        data, targets = get_batch(train_data, i, seq_len=seq_len)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden, device)
        loss = criterion(output.view(-1, number_tokens), targets.long())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        for p in model.parameters():
            p.data.add_(-learning_rate, p.grad.data) # Is this just Stochastic Gradient Descent?

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // seq_len, learning_rate,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(model, corpus, criterion, batch_size = 25, seq_len = 35):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.vocab)
    data_source = batchify(corpus.valid,batch_size)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i, seq_len=seq_len)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets.long()).item()
    return total_loss / (len(data_source) - 1)
