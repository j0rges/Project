# Plot the perplexity of models against their performance on Linzen's
# number agreement task. Different markers for old and new model.
import os, pickle
import matplotlib.pyplot as plt
from training_curves import get_dirs

dir_new = 'train_data'
dir_old = 'train_data/normal'

dirs_new, _ = get_dirs(dir_new)
dirs_old, _ = get_dirs(dir_old)

def load_obj(dir, filename='num_agr_result.pkl'):
    path = os.path.join(dir, filename)
    with open(path, 'rb') as f:
        dict_ = pickle.load(f)
    return dict_

proportion = lambda x: sum(x)/len(x)

dicts_new = [load_obj(dir) for dir in dirs_new]
dicts_old = [load_obj(dir) for dir in dirs_old]

ppl_new = [d['perplexity'] for d in dicts_new]
ppl_old = [d['perplexity'] for d in dicts_old]

num_agr_new = [proportion(d['results']) for d in dicts_new]
num_agr_old = [proportion(d['results']) for d in dicts_old]


plt.plot(ppl_new, num_agr_new, 'rx', label='enhanced model')
plt.plot(ppl_old, num_agr_old, 'bo', label='baseline model')
plt.xlabel('validation perplexity')
plt.ylabel('number agreement accuracy')
plt.legend()
plt.savefig('num_agr.png')
