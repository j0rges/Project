# Plot the final validation perplexity of old and new models against the
# portion of the dataset used for training.
import os
import matplotlib.pyplot as plt
from training_curves import get_dirs, load_dfs, get_attributes

def load_obj(dir, filename='num_agr_result.pkl'):
    path = os.path.join(dir, filename)
    with open(path, 'rb') as f:
        dict_ = pickle.load(f)
    return dict_

proportion = lambda x: sum(x)/len(x)

attributes = ['old_model','dataset_portion']

dir = 'train_data/half'

dirs, names = get_dirs(dir)

attribs = get_attributes(dirs, names, attributes)

dirs_old = [(d,attribs[name]['dataset_portion']) for d,name in zip(dirs,names) \
            if attribs[name]['old_model'] == 'True']
dirs_new = [(d,attribs[name]['dataset_portion']) for d,name in zip(dirs,names) \
            if attribs[name]['old_model'] == 'False']

dicts_new = [(load_obj(dir), portion) for dir, portion in dirs_new]
dicts_old = [(load_obj(dir), portion) for dir, portion in dirs_old]

dicts_new = sorted(dicts_new, key = lambda x: x[1])
dicts_old = sorted(dicts_old, key = lambda x: x[1])

old = [proportion(d['results']) for d,_ in dicts_old]
new = [proportion(d['results']) for d,_ in dicts_new]

fname = 'dataset_size.png'
assert len(old)==len(new)
xs = range(len(old))
plt.plot(xs,old,'o--r', label='classic model')
plt.plot(xs,new,'o--b', label='my model')
plt.ylabel('validation perplexity')
plt.xlabel('% of dataset used in training')
plt.xticks(xs, range(90, 29, -10))
plt.legend()
plt.savefig(fname)
