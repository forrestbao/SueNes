#!/usr/bin/env python3

import os
import pickle

from utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from utils import read_text_file

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
        self.randomed_list = random.sample(self.l, self.length)
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
    res = []
    for r in ratios:
        s = add_words(summary, r, random_word_generator)
        res.append((s, r))
    return res
def mutate_summary_delete(summary):
    ratios = [random.random() for _ in range(10)]
    res = []
    for r in ratios:
        s = delete_words(summary, r)
        res.append((s, r, 'del'))
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
        # this generates 20 mutations
        add_pairs = mutate_summary_add(summary, random_word_generator)
        delete_pairs = mutate_summary_delete(summary)
        item['add-pairs'] = add_pairs
        item['delete-pairs'] = delete_pairs
        outdata[key] = item
    with open(WORD_MUTATED_FILE, 'wb') as f:
        pickle.dump(outdata, f)

def preprocess_sent_mutated():
    """
    NOT IMPLEMENTED.
    """
    return


def preprocess_negtive_sampling():
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
