import os, argparse, pickle, sys
import pandas as pd
import matplotlib.pyplot as plt

description_keys = {'batch_size','dropout','hidden_size','clip_grad'}

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str)
parser.add_argument('--logs', nargs='+')
parser.add_argument('--description')
parser.add_argument('--description-only', action='store_true')

def plot_curves(df_dict, fname='plot.png', time=True):
    """ if time = True plot wrt time, otherwise, plt wrt epoch """
    index = 'time'
    column = 'perplexity'
    for name, df in df_dict.items():
        plt.plot(df[index], df[column], label = name)
    plt.legend()
    plt.xlabel('time (mins)')
    plt.ylabel(column)
    plt.savefig(fname)

def relative_time(df):
    """ Make the time be 0 at the beginning of training,
        and turn it into minutes """
    df.time = (df.time - df.time[0])/60
    return df[1:]

def load_dfs(dirs, names, file='valid.csv'):
    assert len(dirs) == len(names)
    dfs = dict()
    for name, dir in zip(names, dirs):
        filename = os.path.join(dir,file)
        dfs[name] = relative_time(pd.read_csv(filename))
    return dfs

def get_dirs(directory, names=[]):
    if len(names)==0:
        names = os.listdir(directory)
    dirs = []
    for name in names:
        dirs.append(os.path.join(directory, name))
    return dirs, names

def descriptions(dirs, file_to='descriptions.txt', filename='description.txt'):
    with open(file_to, 'w') as fout:
        for dir in dirs:
            file_in = os.path.join(dir, filename)
            fout.write(dir + '\n')
            with open(file_in, 'r') as fin:
                for line in fin:
                    key = line.split(':')[0]
                    if key in description_keys:
                        fout.write(line)
            fout.write('\n')


if __name__=='__main__':
    args = parser.parse_args()

    directory = args.directory
    if args.logs:
        dirs, names = get_dirs(directory, args.logs)
    else:
        dirs, names = get_dirs(directory)
    if args.description:
        descriptions(dirs, file_to=args.description)
    if not args.description_only:
        df_dict = load_dfs(dirs, names)
        plot_curves(df_dict)
