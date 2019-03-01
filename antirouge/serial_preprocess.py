import os
import pickle
import shutil

import numpy as np
import glob

from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from antirouge.utils import read_text_file, sentence_split
from antirouge.utils import dict_pickle_read, dict_pickle_read_keys, dict_pickle_write

from antirouge.embedding import sentence_embed, sentence_embed_reset

import random
import tensorflow as tf

from antirouge.preprocessing import get_art_abs, embed_keep_shape

from antirouge.config import *

SERIAL_BATCH_SIZE = 1000

def serial_process_story():
    """Save as a list. [(article, summary) ...]"""
    # 92,579 stories
    stories = os.listdir(CNN_TOKENIZED_DIR)
    ct = 0
    out_dir = os.path.join(SERIAL_DIR, 'story')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res = []
    fname_ct = 0
    for key in stories:
        ct += 1
        story_file = os.path.join(CNN_TOKENIZED_DIR, key)
        article, summary = get_art_abs(story_file)
        res.append((key, article, summary))
        if ct % SERIAL_BATCH_SIZE == 0:
            fname_ct += 1
            fname = os.path.join(out_dir, "%s.pkl" % fname_ct)
            print('writing %s stories to %s' % (ct, fname))
            with open(fname, 'wb') as f:
                pickle.dump(res, f)
            res = []

def glob_sorted(pattern):
    """Sort according to the number in the filename."""
    return sorted(glob.glob(pattern), key=lambda f:
                  int(''.join(filter(str.isdigit, f))))

def serial_process_embed(embedder, reset_interval=None):
    assert embedder in ['USE', 'USE-Large', 'InferSent']
    out_dir = os.path.join(SERIAL_DIR, embedder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # for each file in story folder, generate the embedding
    story_folder = os.path.join(SERIAL_DIR, 'story')
    all_files = glob_sorted(story_folder + '/*')
    # TODO apply batch?
    ct = 0
    for fname in all_files:
        ct += 1
        out_fname = os.path.join(out_dir, os.path.basename(fname))
        if os.path.exists(out_fname):
            print('%s already exists. Skip.' % out_fname)
            continue
        with open(fname, 'rb') as f:
            stories = pickle.load(f)
            keys = [story[0] for story in stories]
            articles = [story[1] for story in stories]
            summaries = [story[2] for story in stories]
            print('articles: ', len(articles))
            print('split sentences ..')
            # sentence split
            articles = [sentence_split(a) for a in articles]
            summaries = [sentence_split(s) for s in summaries]
            assert len(articles) == len(summaries)
            # encode
            print('encoding sentences ..')
            embedded = embed_keep_shape(articles + summaries, embedder)
            articles = embedded[:len(articles)]
            summaries = embedded[len(articles):]
            # articles = embed_keep_shape(articles, embedder)
            # summaries = embed_keep_shape(summaries, embedder)
            # list(zip([1,2,3], [4,5,6], [7,8,9]))
            obj = list(zip(keys, articles, summaries))
            print('writing to %s ..' % out_fname)
            with open(out_fname, 'wb') as f:
                pickle.dump(obj, f)
            if reset_interval is not None and ct % reset_interval == 0:
                print('reseting tf graph ..')
                sentence_embed_reset()
                tf.reset_default_graph()

def create_tokenizer():
    # read all stories
    story_folder = os.path.join(SERIAL_DIR, 'story')
    all_files = glob_sorted(story_folder + '/*')
    all_text = []
    for fname in all_files:
        with open(fname, 'rb') as f:
            stories = pickle.load(f)
            keys = [story[0] for story in stories]
            articles = [story[1] for story in stories]
            summaries = [story[2] for story in stories]
            all_text.extend(articles)
            all_text.extend(summaries)
    tokenizer = create_tokenizer_from_texts(all_text)
    save_tokenizer(tokenizer)


def __test():
    serial_process_story()
    if True:
        serial_process_embed('USE', 3)
        print('=' * 20)
        sentence_embed_reset()
        tf.reset_default_graph()
        serial_process_embed('USE-Large', 1)
        print('=' * 20)
        sentence_embed_reset()
        tf.reset_default_graph()
        serial_process_embed('InferSent')
