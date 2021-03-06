# Plot training curves of different training instances saved in a directory.
# Also serves as a small library of useful functions for my plots.
import os, argparse, pickle, sys
import pandas as pd
import matplotlib.pyplot as plt

description_keys = {'dropout','hidden_size','clip_grad',
                    'dataset_portion', 'old_model'}

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str)
parser.add_argument('--logs', type=str, nargs='+')
parser.add_argument('--description', type=str, help='filename of description'
                    'file if one is required.')
parser.add_argument('--description-only', action='store_true')
parser.add_argument('--plot-name', type=str, default='plot.png')

def plot_curves(df_dict, fname='plot.png', x_axis='time'):
    column = 'perplexity'
    for name, df in df_dict.items():
        plt.plot(df[x_axis], df[column], label = name)
    plt.legend()
    plt.xlabel(x_axis)
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

def check_dir(path, test_file='valid.csv'):
    if os.path.isdir(path):
        # check test file is in the directory
        if test_file in os.listdir(path):
            return True
    return False

def get_dirs(directory, names=[], test_file='valid.csv'):
    if len(names)==0:
        names = os.listdir(directory)
    dirs = []
    for name in names.copy():
        path = os.path.join(directory, name)
        if check_dir(path, test_file):
            dirs.append(path)
        else:
            names.remove(name)
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

def get_attributes(dirs, names, attributes, filename='description.txt'):
    dict = {}
    for dir,name in zip(dirs,names):
        file_in = os.path.join(dir, filename)
        dict[name] = {}
        with open(file_in, 'r') as fin:
            for line in fin:
                line = line.split(':')
                key = line[0]
                if key in attributes:
                    dict[name][key] = line[1].strip()

    return dict

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
        plot_curves(df_dict, args.plot_name, 'epoch')
