import pandas as pd
import pickle
from training_curves import get_dirs

GOLD_PATH='num_agr/subj_agr_filtered.gold'
RESULTS_FILE='num_agr_result.pkl'

def load_results(dir, filename='num_agr_result.pkl'):
    path = os.path.join(dir, filename)
    with open(path, 'rb') as f:
        dict_ = pickle.load(f)
    return dict_['results']


def results_dataframe(dir):
    dirs, names = get_dirs(dir, test_file=RESULTS_FILE)
    # Load the information about the sentences as a dataframe
    gold = pd.read_csv(GOLD_PATH, delimiter='\t',
                names=['context','right','wrong','attractors'])
    # Load the results for each instance trained.
    results = [load_results(dir) for dir in dirs]
    # Add the results to the dataframe
    for name,result in zip(names,results):
        gold['name'] = result
    return gold, names

def num_attractors_df(gold_df, names):
    new_df = gold_df.[names + ['attractors']].groupby('attractors')\
                    .aggregate(['sum','count'])
    new_df.swaplevel(axis=-1)
    indices = list(zip(['proportion']*7,names))
    # Get the proportion of correct answers for each number of attractors.
    new_df[indices] = new_df['sum']/new_df['count']
    return new_df.swaplevel(axis=-1)

proportion = lambda x: sum(x)/len(x)
def performance_df(gold_df, names):
    performances = [proportion(gold_df[name]) for name in names]
    new_df = pd.DataFrame(data=[performances],columns=names)
    return new_df
