import pandas as pd
import matplotlib.pyplot as plt
import pickle, os
from training_curves import get_dirs

def nonce_gold(path):
    nonce_df = pd.read_csv(path, delimiter='\t')
    gold_df = pd.DataFrame()
    gold_df[['context','right','type']] = nonce_df.loc[nonce_df['class'] == 'correct',
                                    ['len_context', 'form', 'type']].reset_index(drop=True)
    gold_df[['wrong','attractors']] = nonce_df.loc[nonce_df['class'] == 'wrong',
                                    ['form', 'n_attr']].reset_index(drop=True)
    return gold_df

def load_results(dir, filename='num_agr_result.pkl'):
    """ Extract the results of the number agreement task from the directory.
        filename is the name of the file in which they were stored.
    """
    path = os.path.join(dir, filename)
    with open(path, 'rb') as f:
        dict_ = pickle.load(f)
    return dict_['results']


def results_dataframe(dir, gold_path = 'num_agr/subj_agr_filtered.gold',
                            file_path = 'num_agr_result.pkl', nonce=False):
    """ Create a dataframe with information of each test sentence for the number
        agreement task, and whether it was predicted right by each saved model
        in dir with stored results.
    """
    dirs, names = get_dirs(dir, test_file=file_path)
    # Load the information about the sentences as a dataframe
    if nonce:
        gold = nonce_gold(gold_path)
    else:
        gold = pd.read_csv(gold_path, delimiter='\t',
                    names=['context','right','wrong','attractors'])
    # Load the results for each instance trained.
    results = [load_results(dir) for dir in dirs]
    # Add the results to the dataframe
    for name,result in zip(names,results):
        gold[name] = result
    return gold, names

def num_attractors_df(gold_df, names):
    """ names = name of columns with results (abstractly represent a model).
        gold_df = a dataframe with columns names and 'attractors'.
        returns dataframe with attractors as index and the performance of each
        model.
    """
    new_df = gold_df[names + ['attractors']].groupby('attractors')\
                    .aggregate(['sum','count'])
    new_df = new_df.swaplevel(axis=1)
    indices = list(zip(['proportion']*len(names),names))
    # Get the proportion of correct answers for each number of attractors.
    new_df[indices] = new_df['sum']/new_df['count']
    return new_df.swaplevel(axis=1)

proportion = lambda x: sum(x)/len(x)

def performance_df(gold_df, names):
    """ Return dataframe with overall performance for each of names """
    performances = [proportion(gold_df[name]) for name in names]
    new_df = pd.DataFrame(data=[performances],columns=names)
    return new_df

def get_stats(df, column='proportion', swaplevels=True):
    """ return the mean and standard deviation of for columns with column as a
        level (assumes multiindex columns).
    """
    if swaplevels:
        df = df.swaplevel(axis=1)
    values = df[column]
    return values.mean(axis=1).to_numpy(), values.std(axis=1).to_numpy()

def plot_num_attractors(attractors_new, attractors_old,
                        plot_name='acc_attractors.png'):
    """ Plot and save plot of variation of performance and corresponding std for
        input dataframes, as the number of attractors changes. Returns the
        matplotlib.pyplot.axes with the plot.
    """
    plt.rcParams.update({'font.size': 13})
    # Get the index and check
    xs = list(attractors_new.index)
    assert list(attractors_old.index) == xs
    means_new, stds_new = get_stats(attractors_new)
    means_old, stds_old = get_stats(attractors_old)
    # Make the plot
    fig, ax = plt.subplots()
    ax.errorbar(xs, means_new*100, fmt='o-r', yerr=stds_new*100,
                label='enhanced model', capsize = 5)
    ax.errorbar(xs, means_old*100, fmt='o-b', yerr=stds_old*100,
                label='baseline model', capsize = 5)
    ax.set_xticks(xs)
    ax.set_xlabel('number of attractors')
    ax.set_ylabel('accuracy')
    ax.grid(axis='y')
    ax.legend()
    plt.savefig(plot_name)
    return ax

def make_plot(old_dir, new_dir, gold_path = 'num_agr/subj_agr_filtered.gold',
             file_path = 'num_agr_result.pkl', nonce=False):
    new_df, new_names = results_dataframe(new_dir, gold_path, file_path, nonce)
    old_df, old_names = results_dataframe(old_dir, gold_path, file_path, nonce)

    new_df = num_attractors_df(new_df, new_names)
    old_df = num_attractors_df(old_df, old_names)

    return plot_num_attractors(new_df, old_df)
