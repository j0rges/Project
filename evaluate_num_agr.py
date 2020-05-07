# Test number agreeement task on a pre-trained model.
import argparse, os, pickle, torch
import pandas as pd
import numpy as np
from data_loader import get_batch
from utils import load_model_corpora, load_checkpoint

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, default='num_agr/checkpoint.pkl',
        help='path to the checkpoint.')
parser.add_argument('--gold-file', type=str, default='num_agr/subj_agr_filtered.gold',
        help='path to file containing context size, right target and wrong target.')
parser.add_argument('--text-file', type=str, default='num_agr/subj_agr_filtered.text',
        help='path to file containing the sentences.')
parser.add_argument('--nonce', action='store_true',
        help='if provided, indicates the dataset is')

class Word2idx():
    def __init__(self, vocab):
        self.vocab = vocab
    def __call__(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab['<unk>']

def tokenize(path, word2idx):
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf8") as f:
        idss = []
        lengths = []
        for line in f:
            words = line.split()
            lengths.append(len(words))
            ids = [word2idx(word) for word in words]
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)
    return ids, lengths

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def result_(row, logits, word2idx):
    # 1 represents that it succeded at the number agreement task, 0
    # represents it didn't.
    logits = logits[row['idx'],0]
    right = word2idx(row['right'])
    wrong = word2idx(row['wrong'])
    if logits[right] > logits[wrong]:
        return 1
    else:
        return 0

def nonce_gold(path):
    nonce_df = pd.read_csv(path, delimiter='\t')
    gold_df = pd.DataFrame()
    gold_df[['context','right']] = nonce_df.loc[nonce_df['class'] == 'correct',
                                    ['len_context', 'form']].reset_index(drop=True)
    gold_df[['wrong','attractors']] = nonce_df.loc[nonce_df['class'] == 'wrong',
                                    ['form', 'n_attr']].reset_index(drop=True)
    return gold_df

def main(arguments):
    # Get the data we need from the checkpoint
    model, corpora = load_model_corpora(arguments.checkpoint)
    # load the sentences
    word2idx = Word2idx(corpora.vocab)
    sentences, lengths = tokenize(arguments.text_file, word2idx)
    lengths = np.cumsum([0] + lengths[:-1])
    if arguments.nonce:
        gold = nonce_gold(arguments.gold_file)
    else:
        # load the number agreement data, which should be tab sepparated.
        gold = pd.read_csv(arguments.gold_file, delimiter='\t',
                    names=['context','right','wrong','attractors'])
    # Get the location of the target verbs.
    gold['idx'] = gold['context'] + lengths
    # Get the predictions.
    model.eval()
    sentences = batchify(sentences, 1)
    hidden = model.init_hidden(1)
    input, _ = get_batch(sentences, 0, len(sentences))
    output, hidden = model(input, hidden)
    results = gold.apply(lambda x: result_(x, output, word2idx), axis=1)
    checkpoint = load_checkpoint(arguments.checkpoint)
    return results, checkpoint['valid_loss']

if __name__ == "__main__":
    arguments = parser.parse_args()
    results, _ = main(arguments)
    print(sum(results), len(results), sum(results)/len(results))
