# Plot the final validation perplexity of old and new models against the
# portion of the dataset used for training.
import os, pickle, argparse
import matplotlib.pyplot as plt
import numpy as np
from training_curves import get_dirs, load_dfs, get_attributes

attributes = ['old_model','dataset_portion']
dir = 'train_data/half'

parser = argparse.ArgumentParser()
parser.add_argument('--measure', default='perplexity', choices=['num_agr','perplexity'])
parser.add_argument('--complete-baseline', type=str)
parser.add_argument('--complete-enhanced', type=str)
args = parser.parse_args()

def load_obj(dir, filename='num_agr_result.pkl'):
    path = os.path.join(dir, filename)
    with open(path, 'rb') as f:
        dict_ = pickle.load(f)
    return dict_

proportion = lambda x: sum(x)/len(x)

def accuracies(dirs, names, filename='num_agr_result.pkl'):
    results = [load_obj(dir)['results'] for dir in dirs]
    return [proportion(result) for result in results]

def perplexities(dirs, names, filename='num_agr_result.pkl'):
    return [load_obj(dir)['perplexity'] for dir in dirs]

dirs, names = get_dirs(dir)
if args.measure == 'num_agr':
    values = accuracies(dirs, names)
else:
    values = perplexities(dirs, names)

if args.complete_baseline:
    assert not args.complete_enhanced is None
    complete_b = np.array(perplexities(*get_dirs(args.complete_baseline)))
    complete_e = np.array(perplexities(*get_dirs(args.complete_enhanced)))
    complete = True
else:
    complete = False

attribs = get_attributes(dirs, names, attributes)
portion_values = [attribs[name]['dataset_portion'] for name in names]

old = [attribs[name]['old_model'] == 'True' for name in names]
new = [not val for val in old]

# Separate accuracies and portions into old and new
new_pairs = [(a,b) for a,b,c in zip(values, portion_values, new) if c]
old_pairs = [(a,b) for a,b,c in zip(values, portion_values, old) if c]

if complete:
    new_pairs.append((complete_e.mean(), '1.0'))
    old_pairs.append((complete_b.mean(), '1.0'))

# Sort the accuracies based on the portion used in training.
new_pairs = sorted(new_pairs, key = lambda x: x[1])
old_pairs = sorted(old_pairs, key = lambda x: x[1])

def unzip(array, i):
    return [tuple_[i] for tuple_ in array]

portions = unzip(new_pairs, 1)

new_accs = unzip(new_pairs, 0)
old_accs = unzip(old_pairs, 0)

fname = 'dataset_size.png'

# Make the plot.
xs = range(len(portions))
plt.plot(xs,old_accs,'o--b', label='baseline model')
plt.plot(xs,new_accs,'o--r', label='enhanced model')
plt.ylabel('number agreement accuracy')
plt.xlabel('portion of dataset used in training')
plt.xticks(xs, portions)
plt.legend()
plt.savefig(fname)
