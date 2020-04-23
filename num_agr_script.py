# Evaluate and store in a file the number agreement results.
import os, math, pickle, argparse
from training_curves import get_dirs
from evaluate_num_agr import main as num_agr

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str)

dir = parser.parse_args().directory

dirs, names = get_dirs(dir)

class FakeArgs():
    def __init__(self, checkpoint, text_file, gold_file):
        self.checkpoint = checkpoint
        self.text_file = text_file
        self.gold_file = gold_file

arguments = FakeArgs('checkpoint.pkl','num_agr/subj_agr_filtered.text',
                     'num_agr/subj_agr_filtered.gold')
for dir in dirs:
    arguments.checkpoint = os.path.join(dir,'checkpoint.pkl')
    path = os.path.join(dir,'num_agr_result.pkl')
    results, loss = num_agr(arguments)
    to_save = {'results': results, 'perplexity': math.exp(loss)}
    with open(path, 'wb') as f:
        pickle.dump(to_save, f)
