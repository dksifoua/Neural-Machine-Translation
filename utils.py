import re
import tqdm
import string
import pickle
from unicodedata import normalize

def read_file(filepath):
    try:
        with open(filepath, mode='rb') as file:
            content = file.readlines()
        return content
    except:
        raise Error(f'File {filepath} doesn\'t exist')
        
def clean_lines(lines):
    cleaned = []
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    for line in tqdm.tqdm(lines):
        line = line.decode('utf-8', 'ignore')
        line = line.strip()
        # normalize unicode characters
        line = normalize('NFKD', line)
        # Delete multiple spaces
        line = re.sub(' +', ' ', line)
        cleaned.append(line)
    return cleaned

def save(data, filename):
    pickle.dump(data, open(filename, mode='wb'))
    print('Saved: %s' % filename)
    
def load(filename):
    return pickle.load(open(filename, mode='rb'))


class AverageMeter:
    
    def __init__(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.
        
    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.
        
    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count