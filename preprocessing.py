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
        # FIXME I should probably sample uniformly, with return
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
    
def USE_encode_keep_shape(v):
    """This will create sentence embedder!
    """
    use_embedder = SentenceEmbedder()
    flattened = collect(v)
    shape = get_shape(v)
    print('embedding', len(flattened), 'sentences ..')
    embedding_flattened = use_embedder.embed(flattened)
    embedding, _ = restore_shape(embedding_flattened, 0, shape)
    return embedding
def USE_Large_encode_keep_shape(v):
    """This will create sentence embedder!
    """
    use_embedder = SentenceEmbedderLarge()
    flattened = collect(v)
    shape = get_shape(v)
    print('embedding', len(flattened), 'sentences ..')
    embedding_flattened = use_embedder.embed(flattened)
    embedding, _ = restore_shape(embedding_flattened, 0, shape)
    return embedding


def InferSent_encode_keep_shape(v):
    """This will create sentence embedder!
    """
    embedder = InferSentEmbedder()
    flattened = collect(v)
    shape = get_shape(v)
    print('embedding', len(flattened), 'sentences ..')
    embedding_flattened = embedder.embed(flattened)
    embedding, _ = restore_shape(embedding_flattened, 0, shape)
    return embedding

def preprocess_InferSent_story(num):
    """
    This function needs to be called multiple times.

    Currently setting num to 1000 should work

    - 1000 (40,000 sents): 30s
    - 10000 (382,000): out-of-memory
    - 5000: 2min about 10G memory
    """
    if not os.path.exists(INFERSENT_DIR):
        os.makedirs(INFERSENT_DIR)
    story_file = os.path.join(INFERSENT_DIR, 'story.pickle')
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
    if os.path.exists(story_file):
        with open(story_file, 'rb') as f:
            encoded_stories = pickle.load(f)
    else:
        encoded_stories = {}
    ct = 0
    print('Encoded stories:', len(encoded_stories))
    keys = []
    to_encode = []
    for key in stories.keys():
        if key not in encoded_stories:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            story = stories[key]
            article = story['article']
            summary = story['summary']
            to_encode.append(article)
            to_encode.append(summary)
    # encode
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    # this embedding should be (21, 512)
    encoded = InferSent_encode_keep_shape(to_encode_array)
    [1,2,3,4][1::2]
    articles = encoded[0::2]
    summaries = encoded[1::2]
    for key,a,s in zip(keys, articles, summaries):
        item = {}
        item['article'] = a
        item['summary'] = s
        encoded_stories[key] = item
    # write back
    # write to a new file
    tmp_file = os.path.join(INFERSENT_DIR, 'story-tmp.pickle')
    with open(tmp_file, 'wb') as f:
        pickle.dump(encoded_stories, f)
    shutil.copyfile(tmp_file, story_file)
    
def preprocess_USE_story(num):
    """
    This function needs to be called multiple times.

    Currently setting num to 1000 should work

    - 1000 (40,000 sents): 30s
    - 10000 (382,000): out-of-memory
    - 5000: 2min about 10G memory
    """
    if not os.path.exists(USE_DAN_DIR):
        os.makedirs(USE_DAN_DIR)
    story_file = os.path.join(USE_DAN_DIR, 'story.pickle')
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
    if os.path.exists(story_file):
        with open(story_file, 'rb') as f:
            encoded_stories = pickle.load(f)
    else:
        encoded_stories = {}
    ct = 0
    print('Encoded stories:', len(encoded_stories))
    keys = []
    to_encode = []
    for key in stories.keys():
        if key not in encoded_stories:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            story = stories[key]
            article = story['article']
            summary = story['summary']
            to_encode.append(article)
            to_encode.append(summary)
    # encode
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    # this embedding should be (21, 512)
    encoded = USE_encode_keep_shape(to_encode_array)
    [1,2,3,4][1::2]
    articles = encoded[0::2]
    summaries = encoded[1::2]
    for key,a,s in zip(keys, articles, summaries):
        item = {}
        item['article'] = a
        item['summary'] = s
        encoded_stories[key] = item
    # write back
    # write to a new file
    tmp_file = os.path.join(USE_DAN_DIR, 'story-tmp.pickle')
    with open(tmp_file, 'wb') as f:
        pickle.dump(encoded_stories, f)
    shutil.copyfile(tmp_file, story_file)


def preprocess_USE_negative(num):
    """
    This function needs to be called multiple times.
    - 1000 (7524 sents): 20s
    - 10000 (75236 sents): 1min
    - can be all actually: 100,000: (613,000 sents)
    """
    if not os.path.exists(USE_DAN_DIR):
        os.makedirs(USE_DAN_DIR)
    outfile = os.path.join(USE_DAN_DIR, 'negative.pickle')
    with open(NEGATIVE_SAMPLING_FILE, 'rb') as f:
        stories = pickle.load(f)
    if os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            encoded_stories = pickle.load(f)
    else:
        encoded_stories = {}
    ct = 0
    print('Encoded stories:', len(encoded_stories))
    keys = []
    to_encode = []
    for key in stories.keys():
        if key not in encoded_stories:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            fake_summaries = stories[key]
            to_encode.extend(fake_summaries)
    # encode
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    # this embedding should be (21, 512)
    encoded = USE_encode_keep_shape(to_encode_array)
    # encoded contains 5 per entry, so len(encoded) should be 5 * len(keys)
    assert(len(encoded) == len(keys)*5)
    # np.split(np.array([1,2,3,4,5,6,7,8,9,10]), 2)
    splits = np.split(np.array(encoded), len(keys))
    for key,e in zip(keys, splits):
        encoded_stories[key] = e
    # write back
    tmp_file = os.path.join(USE_DAN_DIR, 'negative-tmp.pickle')
    with open(tmp_file, 'wb') as f:
        pickle.dump(encoded_stories, f)
    shutil.copyfile(tmp_file, outfile)

def preprocess_InferSent_negative(num):
    """
    This function needs to be called multiple times.
    - 1000 (7524 sents): 20s
    - 10000 (75236 sents): 1min
    - can be all actually: 100,000: (613,000 sents)
    """
    if not os.path.exists(INFERSENT_DIR):
        os.makedirs(INFERSENTDIR)
    outfile = os.path.join(INFERSENT_DIR, 'negative.pickle')
    with open(NEGATIVE_SAMPLING_FILE, 'rb') as f:
        stories = pickle.load(f)
    if os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            encoded_stories = pickle.load(f)
    else:
        encoded_stories = {}
    ct = 0
    print('Encoded stories:', len(encoded_stories))
    keys = []
    to_encode = []
    for key in stories.keys():
        if key not in encoded_stories:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            fake_summaries = stories[key]
            to_encode.extend(fake_summaries)
    # encode
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    # this embedding should be (21, 512)
    encoded = InferSent_encode_keep_shape(to_encode_array)
    # encoded contains 5 per entry, so len(encoded) should be 5 * len(keys)
    assert(len(encoded) == len(keys)*5)
    # np.split(np.array([1,2,3,4,5,6,7,8,9,10]), 2)
    splits = np.split(np.array(encoded), len(keys))
    for key,e in zip(keys, splits):
        encoded_stories[key] = e
    # write back
    tmp_file = os.path.join(INFERSENT_DIR, 'negative-tmp.pickle')
    with open(tmp_file, 'wb') as f:
        pickle.dump(encoded_stories, f)
    shutil.copyfile(tmp_file, outfile)

def preprocess_USE_mutated(num):
    """
    This function needs to be called multiple times.
    - 1000 (7524 sents): 20s
    - 10000 (75236 sents):
    - can be all actually: 100,000: 
    """
    # TODO Not implemented.
    print('Not implemented.')
    exit(1)
    if not os.path.exists(USE_DAN_DIR):
        os.makedirs(USE_DAN_DIR)
    outfile = os.path.join(USE_DAN_DIR, 'mutated.pickle')
    with open(WORD_MUTATED_FILE, 'rb') as f:
        stories = pickle.load(f)
    if os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            encoded_stories = pickle.load(f)
    else:
        encoded_stories = {}
    ct = 0
    print('Encoded stories:', len(encoded_stories))
    keys = []
    to_encode = []
    for key in stories.keys():
        if key not in encoded_stories:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            add_pairs = stories[key]['add-pairs']
            delete_pairs = stories[key]['delete-pairs']
            fake_summaries = [p[0] for p in (add_pairs + delete_pairs)]
            to_encode.extend(fake_summaries)
    # encode
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    # this embedding should be (21, 512)
    encoded = USE_encode_keep_shape(to_encode_array)
    # encoded contains 5 per entry, so len(encoded) should be 5 * len(keys)
    assert(len(encoded) == len(keys)*20)
    # np.split(np.array([1,2,3,4,5,6,7,8,9,10]), 2)
    splits = np.split(np.array(encoded), len(keys))
    for key,e in zip(keys, splits):
        encoded_stories[key] = e
    # write back
    with open(outfile, 'wb') as f:
        pickle.dump(encoded_stories, f)

        
def use_pregen():
    """Pre-generate the uae for data folder.
    """
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    data_dir = os.path.join(cnndm_dir, 'hebi-sample-10000')
    hebi_uae_dir = os.path.join(cnndm_dir, 'hebi-uae')
    # For each folder (with hash) in hebi_dir, check if hebi_uae_dir
    # contains this folder or not. If not, parse the article and
    # summaries into data.
    stories = os.listdir(data_dir)

    # printing out current progress
    finished_stories = os.listdir(hebi_uae_dir)
    print('total stories:', len(stories))
    print('finished:', len(finished_stories))

    print('creating UAE instance ..')
    use_embedder = SentenceEmbedder()

    ct = 0

    for s in stories:
        data = {}
        to_encode = []
        scores = []
        story_dir = os.path.join(data_dir, s)
        story_uae_file = os.path.join(hebi_uae_dir, s)
        if not os.path.exists(story_uae_file):
            print('processing', s, '..')
            ct += 1
            if ct % 50 == 0:
                # This function returns after processing 50 stories,
                # due to memory reason.
                print('reaches 50 stories, existing ..')
                return
            
            # create the embedding
            article_file = os.path.join(story_dir, 'article.txt')
            summary_file = os.path.join(story_dir, 'summary.json')
            with open(article_file) as f:
                article = f.read()
                to_encode.append(article)
            with open(summary_file) as f:
                j = json.load(f)
                for summary,score,_ in j:
                    to_encode.append(summary)
                    scores.append(score)
            # now, encode to_encode
            # to_encode should have 21 paragraphs
            # split into sentences
            to_encode_array = [sentence_split(a) for a in to_encode]
            def get_shape(vv):
                return [len(v) for v in vv]
            def flatten(vv):
                res = []
                for v in vv:
                    res.extend(v)
                return res
            def restructure(v, shape):
                if not shape:
                    return []
                l = shape[0]
                return [v[:l]] + restructure(v[l:], shape[1:])
            shape = get_shape(to_encode_array)
            flattened = flatten(to_encode_array)
            print('embedding', len(flattened), 'sentences ..')
            embedding_flattened = use_embedder.embed(flattened)
            embedding = restructure(embedding_flattened, shape)
            # this embedding should be (21, 512)
            data = {}
            #(numsent, 512)
            data['article'] = embedding[0]
            data['summary'] = embedding[1:]
            data['score'] = scores
            with open(story_uae_file, 'wb') as f:
                pickle.dump(data, f)

def preprocess_USE_Large_story(num):
    """
    This function needs to be called multiple times.

    Currently setting num to 1000 should work

    - 1000 (40,000 sents): 30s
    - 10000 (382,000): out-of-memory
    - 5000: 2min about 10G memory
    """
    if not os.path.exists(USE_TRANSFORMER_DIR):
        os.makedirs(USE_TRANSFORMER_DIR)
    story_file = os.path.join(USE_TRANSFORMER_DIR, 'story.pickle')
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
    if os.path.exists(story_file):
        with open(story_file, 'rb') as f:
            encoded_stories = pickle.load(f)
    else:
        encoded_stories = {}
    ct = 0
    print('Encoded stories:', len(encoded_stories))
    keys = []
    to_encode = []
    for key in stories.keys():
        if key not in encoded_stories:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            story = stories[key]
            article = story['article']
            summary = story['summary']
            to_encode.append(article)
            to_encode.append(summary)
    # encode
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    # this embedding should be (21, 512)
    encoded = USE_Large_encode_keep_shape(to_encode_array)
    [1,2,3,4][1::2]
    articles = encoded[0::2]
    summaries = encoded[1::2]
    for key,a,s in zip(keys, articles, summaries):
        item = {}
        item['article'] = a
        item['summary'] = s
        encoded_stories[key] = item
    # write back
    # write to a new file
    tmp_file = os.path.join(USE_TRANSFORMER_DIR, 'story-tmp.pickle')
    with open(tmp_file, 'wb') as f:
        pickle.dump(encoded_stories, f)
    shutil.copyfile(tmp_file, story_file)


def preprocess_USE_Large_negative(num):
    """
    This function needs to be called multiple times.
    - 1000 (7524 sents): 20s
    - 10000 (75236 sents): 1min
    - can be all actually: 100,000: (613,000 sents)
    """
    if not os.path.exists(USE_TRANSFORMER_DIR):
        os.makedirs(USE_TRANSFORMER_DIR)
    outfile = os.path.join(USE_TRANSFORMER_DIR, 'negative.pickle')
    with open(NEGATIVE_SAMPLING_FILE, 'rb') as f:
        stories = pickle.load(f)
    if os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            encoded_stories = pickle.load(f)
    else:
        encoded_stories = {}
    ct = 0
    print('Encoded stories:', len(encoded_stories))
    keys = []
    to_encode = []
    for key in stories.keys():
        if key not in encoded_stories:
            ct += 1
            if ct % num == 0:
                break
            keys.append(key)
            fake_summaries = stories[key]
            to_encode.extend(fake_summaries)
    # encode
    if not keys:
        print('All encoded!')
    to_encode_array = [sentence_split(a) for a in to_encode]
    # this embedding should be (21, 512)
    encoded = USE_Large_encode_keep_shape(to_encode_array)
    # encoded contains 5 per entry, so len(encoded) should be 5 * len(keys)
    assert(len(encoded) == len(keys)*5)
    # np.split(np.array([1,2,3,4,5,6,7,8,9,10]), 2)
    splits = np.split(np.array(encoded), len(keys))
    for key,e in zip(keys, splits):
        encoded_stories[key] = e
    # write back
    tmp_file = os.path.join(USE_TRANSFORMER_DIR, 'negative-tmp.pickle')
    with open(tmp_file, 'wb') as f:
        pickle.dump(encoded_stories, f)
    shutil.copyfile(tmp_file, outfile)
