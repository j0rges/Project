# Plot the final validation perplexity of old and new models against the
# portion of the dataset used for training.
import os
import matplotlib.pyplot as plt
from training_curves import get_dirs, load_dfs, get_attributes

attributes = ['old_model','dataset_portion']

dir = 'train_data/half'

dirs, names = get_dirs(dir)

dfs = load_dfs(dirs, names)

attribs = get_attributes(dirs, names, attributes)

old = {key:item for key,item in dfs.items() if \
        attribs[key]['old_model'] == 'True'}
new = {key:item for key,item in dfs.items() if \
        attribs[key]['old_model'] != 'True'}

old = sorted([df['perplexity'].to_numpy()[-1] for key,df in old.items()])
new = sorted([df['perplexity'].to_numpy()[-1] for key,df in new.items()])

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
