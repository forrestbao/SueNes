#!/usr/bin/env python3

import os
import pickle
import shutil

import numpy as np

from utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from utils import read_text_file, sentence_split

from embedding import SentenceEmbedder, SentenceEmbedderLarge, InferSentEmbedder

import random

from config import *


def get_art_abs(story_file):
    lines = read_text_file(story_file)
    lines = [line.lower() for line in lines]
    # lines = [fix_missing_period(line) for line in lines]
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    article = ' '.join(article_lines)
    abstract = ' '.join(highlights)
    return article, abstract

class RandomItemGenerator():
    def __init__(self, l):
        self.l = l
        self.length = len(self.l)
        self.reset_cache()
    def reset_cache(self):
        # self.randomed_list = random.sample(self.l, self.length)
        # I should probably sample uniformly, with replacement:
        self.randomed_list = random.choices(self.l, k=self.length)
        self.random_index = 0
    def random_item(self, exclude=[]):
        self.random_index += 1
        if self.random_index >= self.length:
            self.reset_cache()
        res = self.randomed_list[self.random_index]
        if res in exclude:
            # FIXME stack overflow
            return self.random_item(exclude=exclude)
        else:
            return res
        
def delete_words(summary, ratio):
    words = summary.split(' ')
    length = len(words)
    indices = set(random.sample(range(length),
                                int((1 - ratio) * length)))
    return ' '.join([words[i] for i in range(length)
                     if i not in indices])

def replace_words(summary, ratio, random_word_generator):
    words = summary.split(' ')
    length = len(words)
    indices = set([random.randint(0, length)
                   for _ in range(int((1 - ratio) * length))])
    res = []
    for i in range(length):
        if i in indices:
            res.append(random_word_generator.random_item())
        else:
            res.append(words[i])
    return ' '.join(res)

def add_words(summary, ratio, random_word_generator):
    words = summary.split(' ')
    length = len(words)
    indices = set([random.randint(0, length)
                   for _ in range(int((1 - ratio) * length))])
    res = []
    for i in range(length):
        if i in indices:
            res.append(random_word_generator.random_item())
        res.append(words[i])
    return ' '.join(res)

def mutate_summary_add(summary, random_word_generator):
    ratios = [random.random() for _ in range(10)]
    res = {}
    res['text'] = []
    res['label'] = []
    for r in ratios:
        s = add_words(summary, r, random_word_generator)
        res['text'].append(s)
        res['label'].append(r)
    return res
def mutate_summary_delete(summary):
    ratios = [random.random() for _ in range(10)]
    res = {}
    res['text'] = []
    res['label'] = []
    for r in ratios:
        s = delete_words(summary, r)
        res['text'].append(s)
        res['label'].append(r)
    return res

def mutate_summary_replace(summary, random_word_generator):
    ratios = [random.random() for _ in range(10)]
    res = {}
    res['text'] = []
    res['label'] = []
    for r in ratios:
        s = replace_words(summary, r, random_word_generator)
        res['text'].append(s)
        res['label'].append(r)
    return res

def preprocess_story_pickle():
    """Read text, parse article and summary text file into pickle
    {'article': 'xxxxx', 'summary': 'xxxxxxx'}

    """
    # 92,579 stories
    stories = os.listdir(CNN_TOKENIZED_DIR)
    ct = 0
    data = {}
    for key in stories:
        ct += 1
        if ct % 100 == 0:
            print ('--', ct*100)
        story_file = os.path.join(CNN_TOKENIZED_DIR, key)
        item = {}
        article, summary = get_art_abs(story_file)
        item['article'] = article
        item['summary'] = summary
        data[key] = item
    with open(STORY_PICKLE_FILE, 'wb') as f:
        pickle.dump(data, f)

def preprocess_word_mutated():
    """
    From story_pickle to word mutated.
    """
    print('loading article ..')
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
    articles = [s['article'] for s in stories.values()]
    summaries = [s['summary'] for s in stories.values()]
    print('creating tokenizer ..')
    tokenizer = create_tokenizer_from_texts(articles + summaries)
    random_word_generator = RandomItemGenerator(list(tokenizer.word_index.keys()))
    random_word_generator.random_item()
    outdata = {}
    ct = 0
    print('generating mutated summary for', len(stories), 'stories ..')
    for key in stories:
        ct += 1
        if ct % 100 == 0:
            print ('--', ct)
        story = stories[key]
        summary = story['summary']
        item = {}
        # this generates 10 mutations for each setting
        add = mutate_summary_add(summary, random_word_generator)
        delete = mutate_summary_delete(summary)
        replace = mutate_summary_replace(summary, random_word_generator)
        item['add'] = add
        item['delete'] = delete
        item['replace'] = replace
        outdata[key] = item
    with open(WORD_MUTATED_FILE, 'wb') as f:
        pickle.dump(outdata, f)

def test():
    preprocess_word_mutated()

def preprocess_sent_mutated():
    """
    NOT IMPLEMENTED.
    """
    return

def preprocess_negative_sampling():
    """Read all articles and summaries. For each (article + summary) with
    label 1 pair, construct 5 (article + fake summary) pair with label
    0, as a block.
    """
    print('loading article ..')
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
    articles = [s['article'] for s in stories.values()]
    summaries = [s['summary'] for s in stories.values()]
    generator = RandomItemGenerator(summaries)
    outdata = {}
    ct = 0
    print('generating negative sampling for', len(stories), 'stories ..')
    for key in stories:
        ct += 1
        if ct % 100 == 0:
            print ('--', ct)
        story = stories[key]
        summary = story['summary']
        article = story['article']
        outdata[key] = [generator.random_item(exclude=[summary])
                        for _ in range(5)]
    with open(NEGATIVE_SAMPLING_FILE, 'wb') as f:
        pickle.dump(outdata, f)

def shuffle_string(s):
    """Shuffle all words in the string.
    - find all sentence splits
    - shuffle inner sentence
    """
    sents = sentence_split(s)
    res = []
    for sent in sents:
        tokens = sent.split(' ')
        new_sent = ' '.join(random.sample(tokens[:-1], len(tokens)-1) + tokens[-1:])
        res.append(new_sent)
    return ' '.join(res)

def test():
    sentence_split('hello world bababa . yes and no')
    shuffle_string('hello world bababa . yes and no')
    
def preprocess_negative_shuffle():
    """FIXME if sentence is too small, we may end up with really tiny
    change.

    """
    print('loading article ..')
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
    articles = [s['article'] for s in stories.values()]
    summaries = [s['summary'] for s in stories.values()]
    outdata = {}
    ct = 0
    print('generating negative shuffle sampling for', len(stories), 'stories ..')
    for key in stories:
        ct += 1
        if ct % 100 == 0:
            print ('--', ct)
        story = stories[key]
        summary = story['summary']
        article = story['article']
        # shuffle the words
        outdata[key] = [shuffle_string(summary) for _ in range(5)]
    with open(NEGATIVE_SHUFFLE_FILE, 'wb') as f:
        pickle.dump(outdata, f)


# collect into one arry
def collect(v):
    res = []
    for vi in v:
        if type(vi) == list:
            res.extend(collect(vi))
        else:
            res.append(vi)
    return res

def get_shape(v):
    """
    [[[1,2], [3,4,5]], [[6,7], [8,9,10]]]
    [[2,3], [2,3]]
    """
    assert(type(v) is list)
    res = []
    if not v:
        return 1
    if type(v[0]) is not list:
        return len(v)
    for vi in v:
        res.append(get_shape(vi))
    return res

def restore_shape(v, index, shape):
    """v is flat
    """
    res = []
    if type(shape) is not list:
        return v[index:index+shape], index+shape
    for s in shape:
        sub, index = restore_shape(v, index, s)
        res.append(sub)
    return res, index

def test():
    array = [[[1,2], [3,4,5]], [[6,7], [8,9,10]]]
    shape = get_shape(array)
    flat = collect(array)
    restored, index = restore_shape(flat, 0, shape)
    assert(array == restored)
    assert(index == len(flat))

    v = [[['hello', 'world'], ['hello', 'world', 'ok']], [['yes', 'no']]]
    encoded = USE_encode_keep_shape(v)

def embed_keep_shape(v, embedder_name):
    if embedder_name == 'USE':
        embedder = SentenceEmbedder()
    elif embedder_name == 'USE-Large':
        embedder = SentenceEmbedderLarge()
    elif embedder_name == 'InferSent':
        embedder = InferSentEmbedder()
    else:
        print(embedder_name)
        raise Exception('Embedder not specified error.')
    flattened = collect(v)
    shape = get_shape(v)
    print('embedding', len(flattened), 'sentences ..')
    embedding_flattened = embedder.embed(flattened)
    embedding, _ = restore_shape(embedding_flattened, 0, shape)
    return embedding

def load_to_encode(target_name, previously_encoded, num):
    assert(target_name in ['story', 'negative', 'mutated', 'shuffle'])
    target_dict = {'story': STORY_PICKLE_FILE, 'negative':
                   NEGATIVE_SAMPLING_FILE, 'mutated':
                   WORD_MUTATED_FILE, 'shuffle':
                   NEGATIVE_SHUFFLE_FILE}
    plain_file = target_dict[target_name]
    to_encode = []
    with open(plain_file, 'rb') as f:
        plain_stories = pickle.load(f)
    ct = 0
    keys = []
    for key in plain_stories.keys():
        if key not in previously_encoded:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            story = plain_stories[key]
            if target_name == 'story':
                article = story['article']
                summary = story['summary']
                to_encode.append(article)
                to_encode.append(summary)
            elif target_name == 'negative' or target_name == 'shuffle':
                fake_summaries = plain_stories[key]
                to_encode.extend(fake_summaries)
            else:
                add = plain_stories[key]['add']
                delete = plain_stories[key]['delete']
                replace = plain_stories[key]['replace']
                to_encode.extend(add['text'])
                to_encode.extend(delete['text'])
                to_encode.extend(replace['text'])
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    return keys, to_encode_array

def save_encoded(target_name, keys, encoded):
    res = {}
    if target_name == 'story':
        articles = encoded[0::2]
        summaries = encoded[1::2]
        for key,a,s in zip(keys, articles, summaries):
            item = {}
            item['article'] = a
            item['summary'] = s
            res[key] = item
    elif target_name == 'negative' or target_name == 'shuffle':
        splits = np.split(np.array(encoded), len(keys))
        for key,e in zip(keys, splits):
            res[key] = e
    elif target_name == 'mutated':
        splits = np.split(np.array(encoded), len(keys))
        with open(WORD_MUTATED_FILE, 'rb') as f:
            orig_mutated = pickle.load(f)
        for key,e in zip(keys, splits):
            sp = np.split(e, 3)
            
            res[key]['add'] = {}
            res[key]['delete'] = {}
            res[key]['replace'] = {}
            
            res[key]['add']['text'] = sp[0]
            res[key]['delete']['text'] = sp[1]
            res[key]['replace']['text'] = sp[2]
            
            res[key]['add']['label'] = orig_mutated[key]['add']['label']
            res[key]['delete']['text'] = orig_mutated[key]['delete']['label']
            res[key]['replace']['text'] = orig_mutated[key]['replace']['label']
    return res
    
                
def preprocess_sentence_embed(embedder_name, target_name, num):
    assert(embedder_name in ['InferSent', 'USE', 'USE-Large'])
    assert(target_name in ['story', 'negative', 'mutated', 'shuffle'])
    folder_dict = {'InferSent': INFERSENT_DIR, 'USE': USE_DAN_DIR,
                   'USE-Large': USE_TRANSFORMER_DIR}
    folder = folder_dict[embedder_name]
    if not os.path.exists(folder):
        os.makedirs(folder)
    target_file = os.path.join(folder, target_name + '.pickle')
    # TODO find all files in folder with pattern
    # <target_name>-<digit>.pickle
    tmp_file = os.path.join(folder, 'tmp.pickle')
    if os.path.exists(target_file):
        with open(target_file, 'rb') as f:
            previously_encoded = pickle.load(f)
    else:
        previously_encoded = {}
    print('Previously encoded: ', len(previously_encoded))
    # Load what to encode, based on target name
    keys, to_encode = load_to_encode(target_name, previously_encoded, num)
    encoded = embed_keep_shape(to_encode, embedder_name)
    newly_encoded = save_encoded(target_name, keys, encoded)
    # TODO I want to save it to a new file, such that we won't have
    # the overhead of writing big files every time. I shall also
    # provide a procedure to combine all these files.
    # DEBUG
    previously_encoded.update(newly_encoded)
    with open(tmp_file, 'wb') as f:
        pickle.dump(previously_encoded, f)
    shutil.copyfile(tmp_file, target_file)
    return


def test():
    preprocess_sentence_embed('InferSent', 'story', 10)
    preprocess_sentence_embed('InferSent', 'negative', 10)
    preprocess_sentence_embed('USE', 'mutated', 10)
    preprocess_sentence_embed('USE', 'shuffle', 10000)
    preprocess_sentence_embed('USE-Large', 'negative', 10)
    preprocess_sentence_embed('InferSent', 'story', 10)
