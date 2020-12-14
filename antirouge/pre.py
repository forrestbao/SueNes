"""
serial_process_story(in_dir, out_dir, batch_size=100)
serial_process_embed(base_folder, embedder)
"""

import os
import pickle
import shutil
import csv

import numpy as np
import json
import math
import os
import subprocess

import random
import tensorflow as tf


from bs4 import BeautifulSoup

from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from antirouge.utils import sentence_split
from antirouge.utils import dict_pickle_read, dict_pickle_read_keys, dict_pickle_write

from antirouge.embedding import sentence_embed, embed_keep_shape
from antirouge.utils import read_lines, sentence_split, glob_sorted


from antirouge import config
from antirouge.utils import download, unzip

def ensure_corenlp_tokenizer():
    if os.path.exists(config.CORENLP_JAR):
        # FIXME DEBUG
        download('http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip',
                 'stanford-corenlp-full-2018-10-05.zip')
        unzip('stanford-corenlp-full-2018-10-05.zip',
              'stanford-corenlp-full-2018-10-05')
        shutil.copyfile('stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar',
                        config.CORENLP_JAR)
    os.environ['CLASSPATH'] = config.CORENLP_JAR


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using
Stanford CoreNLP Tokenizer"""
    if not os.path.exists(tokenized_stories_dir):
        os.makedirs(tokenized_stories_dir)
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n"
                    % (os.path.join(stories_dir, s),
                       os.path.join(tokenized_stories_dir, s)))
    ensure_corenlp_tokenizer()
    command = ['java',
                'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..."
          % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same
    # number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    print("Successfully finished tokenizing %s to %s.\n"
          % (stories_dir, tokenized_stories_dir))


def get_art_abs(story_file):
    lines = read_lines(story_file)
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


def serial_process_story(in_dir, out_dir, batch_size=100):
    """Save as a list. [(article, summary) ...].

    in_dir CNN_TOKENIZED_DIR, should contains a list of xxx.story, text with
    both article and summary

    out_dir os.path.join(SERIAL_DIR, 'story'), will output 1.pkl

    The important configuration is the total stories to use and the batch size.
    There are totally 92,579 stories. If that's too much, you probably want to
    sample several from it, e.g. 10000. Then, you need to decide a batch size,
    e.g. if 100, then there are 100 batches, and  100 pickle files will be
    created.
    """
    # 92,579 stories
    stories = os.listdir(in_dir)
    # CAUTION FIXME DEBUG using only the first 10000 stories
    stories = stories[:10000]
    ct = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res = []
    fname_ct = 0
    for key in stories:
        ct += 1
        story_file = os.path.join(in_dir, key)
        article, summary = get_art_abs(story_file)
        res.append((key, article, summary))
        if ct % batch_size == 0:
            fname_ct += 1
            fname = os.path.join(out_dir, "%s.pkl" % fname_ct)
            print('writing %s stories to %s' % (ct, fname))
            with open(fname, 'wb') as f:
                pickle.dump(res, f)
            res = []
def process_tsv_folder(folder, embedder, batch_size=100):
    process_tsv(os.path.join(folder, 'train.tsv'), embedder, batch_size)
    process_tsv(os.path.join(folder, 'test.tsv'), embedder, batch_size)
    process_tsv(os.path.join(folder, 'validation.tsv'), embedder, batch_size)


def process_tsv(tsv_file, embedder, batch_size=100):
    """Process new tsv data files"""
    # 1. read train.tsv, validation.tsv, and test.tsv
    # 2. generate
    #    - train_USE.pkl, val_USE.pkl, test_USE.pkl
    #    - train_USE-Large.pkl
    #    - train_InferSent.pkl
    # 3. optionally, split into x.pkl
    articles = []
    summaries = []
    scores = []
    out_folder = os.path.join(os.path.dirname(tsv_file), 
                              os.path.basename(tsv_file) + '_embed')
    print('reading ', tsv_file)
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='	')
        for row in reader:
            article = row[0]
            summary = row[1]
            score = row[2]
            articles.append(article)
            summaries.append(summary)
            scores.append(score)
    print('number of articles: ', len(articles))
    # sentence split
    articles = [sentence_split(a) for a in articles]
    summaries = [sentence_split(s) for s in summaries]
    # dummy keys
    keys = range(len(articles))
    assert len(articles) == len(summaries)
    # encode
    print('encoding sentences ..')
    rg = range(0, len(summaries), batch_size)
    # start = time.time()
    print('total number of pickle files to write: ', len(rg))
    for idx,stidx in enumerate(rg):
        # if idx % 300 == 0:
        #     pass
        out_fname = os.path.join(out_folder, embedder, '{}.pkl'.format(idx))
        if os.path.exists(out_fname):
            print('skipping', out_fname)
            continue
        print('embedding ..')
        batch_a = articles[stidx:stidx + batch_size]
        batch_s = summaries[stidx:stidx + batch_size]
        embedded = embed_keep_shape(batch_a + batch_s, embedder)
        obj = list(zip(range(batch_size), embedded[:batch_size], embedded[batch_size:]))
        
        if not os.path.exists(os.path.dirname(out_fname)):
            os.makedirs(os.path.dirname(out_fname))
        print('writing to %s ..' % out_fname)
        with open(out_fname, 'wb') as f:
            pickle.dump(obj, f)



def serial_process_embed(base_folder, embedder):
    """Process folder/story and write folder/XXX where XXX is the embedder
name.

    base_folder SERIAL_DIR

    """
    assert embedder in ['USE', 'USE-Large', 'InferSent']
    out_dir = os.path.join(base_folder, embedder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # for each file in story folder, generate the embedding
    story_folder = os.path.join(base_folder, 'story')
    all_files = glob_sorted(story_folder + '/*')
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



if __name__ == '__main__':
    serial_process_story(config.CNN_TOKENIZED_DIR,
                            os.path.join(config.CNN_SERIAL_DIR, 'story'))
    serial_process_embed(config.CNN_SERIAL_DIR, 'USE')
    serial_process_embed(config.CNN_SERIAL_DIR, 'USE-Large')
    serial_process_embed(config.CNN_SERIAL_DIR, 'InferSent')
