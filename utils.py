from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
# why keras.preprocessing is not exporting tokenizer_from_json ??
from keras_preprocessing.text import tokenizer_from_json

from config import *

def create_tokenizer_from_texts(texts):
    """Tokenizer needs to fit on the given text. Then, we can use it to
    obtain:

    1. tokenizer.texts_to_sequences (texts)
    2. tokenizer.word_index

    """
    # finally, vectorize the text samples into a 2D integer tensor
    # num_words seems to be not useful at all. I set it to 20,000, but
    # the tokenizer.index_word is still 178,781. I set it to 200,000
    # to be consistent
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer
def save_tokenizer(tokenizer):
    # save tokenizer
    j_str = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(j_str)
def load_tokenizer():
    # load
    with open('tokenizer.json') as f:
        j_str = f.read()
        tokenizer = tokenizer_from_json(j_str)
        return tokenizer
def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def save_data(data, filename):
    (x_train, y_train), (x_val, y_val) = data
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data

def sentence_split(s):
    # FIXME keep the seperator
    res = re.split(r'\.|!|\?', s)
    res = [r.strip() for r in res if r]
    return res

def test():
    test_str = 'hello hello hello . world world ! eh eh eh ? yes yes ... ok ok'
    sentence_split(test_str)
