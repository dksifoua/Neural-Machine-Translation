import re
import unicodedata

def read_file(filepath):
    try:
        with open(filepath, mode='rt', encoding='utf-8') as file:
            content = file.readlines()
        return content
    except:
        raise Error(f'File {filepath} doesn\'t exist')
        
def unicode_to_ascii(s):
    # NFD => Normal Form Decompose
    # Mn => Non Marking Space
    return ''.join(c for c in unicodedata.normalize('NFD', s) \
                    if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    # Transform accented characters into unaccented ones
    s = unicode_to_ascii(s.strip())
    # Replace any of '.', '!', '?' by ' .', ' !', ' ?'. \1 means the 1st bracked group. r is to not consider \1
    s = re.sub(r'([,.!?0-9])', r' \1', s)
    # Remove any character which is not in [^a-zA-Z0-9,.!?]
    s = re.sub(r'[^a-zA-Z0-9,.!?]', r' ', s)
    # Remove a sequence of whitespace characters
    s = re.sub(r'\s+', r' ', s)
    return s.strip()


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