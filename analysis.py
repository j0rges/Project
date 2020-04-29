import pandas as pd
import pickle, os
from training_curves import get_dirs

GOLD_PATH='num_agr/subj_agr_filtered.gold'
RESULTS_FILE='num_agr_result.pkl'

def load_results(dir, filename='num_agr_result.pkl'):
    """ Extract the results of the number agreement task from the directory.
        filename is the name of the file in which they were stored.
    """
    path = os.path.join(dir, filename)
    with open(path, 'rb') as f:
        dict_ = pickle.load(f)
    return dict_['results']


def results_dataframe(dir):
    """ Create a dataframe with information of each test sentence for the number
        agreement task, and whether it was predicted right by each saved model
        in dir with stored results.
    """
    dirs, names = get_dirs(dir, test_file=RESULTS_FILE)
    # Load the information about the sentences as a dataframe
    gold = pd.read_csv(GOLD_PATH, delimiter='\t',
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
