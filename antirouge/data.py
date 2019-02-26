#!/usr/bin/env python3
import pickle
import numpy as np
import os
import json
import re
import math
import random

from keras.preprocessing.sequence import pad_sequences

from utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from utils import read_text_file

from config import *

def prepare_data_using_USE():
    """Return vector of float32. The dimension of X is (?,13,512), Y is
    (?) where ? is the number of articles.
    """
    # data_dir = os.path.join(cnndm_dir, 'hebi-sample-10000')
    hebi_uae_dir = os.path.join(cnndm_dir, 'hebi-uae')
    # I'm going to use just uae dir
    stories = os.listdir(hebi_uae_dir)
    len(stories)
    article_data = []
    summary_data = []
    scores = []
    print('loading USE preprocessed stories ..')
    for s in stories:
        uae_file = os.path.join(hebi_uae_dir, s)
        with open(uae_file, 'rb') as f:
            data = pickle.load(f)
            length = len(data['summary'])
            # the embedding of the article
            article_data.extend([data['article']] * length)
            # a list of embedding of summaries
            summary_data.extend(data['summary'])
            scores.extend(data['score'])
    # pad sequence
    print('padding sequence ..')
    article_data_padded = pad_sequences(article_data,
                                        value=np.zeros(512), padding='post',
                                        maxlen=ARTICLE_MAX_SENT, dtype='float32')
    summary_data_padded = pad_sequences(summary_data,
                                        value=np.zeros(512), padding='post',
                                        maxlen=SUMMARY_MAX_SENT, dtype='float32')
    print('concatenating ..')
    data = np.concatenate((article_data_padded, summary_data_padded), axis=1)

    return shuffle_and_split(data, np.array(scores))

def prepare_data_with_USE(articles, summaries, labels, group):
    print('padding sequence ..')
    article_data_padded = pad_sequences(articles,
                                        value=np.zeros(512), padding='post',
                                        maxlen=ARTICLE_MAX_SENT, dtype='float32')
    summary_data_padded = pad_sequences(summaries,
                                        value=np.zeros(512), padding='post',
                                        maxlen=SUMMARY_MAX_SENT, dtype='float32')
    print('concatenating ..')
    data = np.concatenate((article_data_padded, summary_data_padded), axis=1)

    return shuffle_and_split(data, np.array(labels), group)
    
def prepare_data_with_INFER(articles, summaries, labels, group):
    print('padding sequence ..')
    article_data_padded = pad_sequences(articles,
                                        value=np.zeros(4096), padding='post',
                                        maxlen=ARTICLE_MAX_SENT, dtype='float32')
    summary_data_padded = pad_sequences(summaries,
                                        value=np.zeros(4096), padding='post',
                                        maxlen=SUMMARY_MAX_SENT, dtype='float32')
    print('concatenating ..')
    data = np.concatenate((article_data_padded, summary_data_padded), axis=1)

    return shuffle_and_split(data, np.array(labels), group)

def load_story_keys(size=None):
    """Return a list of keys.
    """
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
        len(stories)
        if not size:
            return list(stories.keys())
        elif size > len(stories):
            print('Error: attempt to load too many')
            exit(1)
        else:
            return random.sample(stories.keys(), size)

def create_tokenizer_by_key(keys):
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
        articles = [stories[key]['article'] for key in keys]
        summaries = [stories[key]['summary'] for key in keys]
        tokenizer = create_tokenizer_from_texts(articles + summaries)
        return tokenizer

def create_tokenizer():
    """FIXME I'm going to train tokenizer just using all the text.
    """
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
        articles = [story['article'] for story in stories.values()]
        summaries = [story['summary'] for story in stories.values()]
        tokenizer = create_tokenizer_from_texts(articles + summaries)
        save_tokenizer(tokenizer)
    
    
def load_word_mutated_data(keys, mode):
    """
    Return (summaries, scores)

    MODE can be 'add', 'del', 'both', 'shuffle'
    """
    with open(WORD_MUTATED_FILE, 'rb') as f:
        mut = pickle.load(f)
        summaries = []
        scores = []
        for key in keys:
            pairs = []
            if mode == 'add':
                pairs = mut[key]['add-pairs']
            elif mode == 'del':
                pairs = mut[key]['delete-pairs']
            elif mode == 'both':
                pairs = mut[key]['delete-pairs'] + mut[key]['add-pairs']
            else:
                print('Error: Use only add, del, both')
                exit(1)
            summaries.append([p[0] for p in pairs])
            scores.append([p[1] for p in pairs])
        return np.array(summaries), np.array(scores)

    

def test_keras_preprocessing():
    tokenizer.texts_to_sequences(['hello world you are awesome', 'you is good person'])
    tokenizer.sequences_to_texts([[10468, 88, 35, 22, 6270], [35, 11, 199, 363]])
    one_hot('hello world you are you awesome', 200)
    return

def shuffle_and_split(features, labels, group):
    """Use interval to control how the split happens. For example
    group=21.

    1. group into groups
    2. shuffle groups, split groups
    3. flatten groups
    4. (optional) shuffle again

    """
    print('splitting by group ..')
    features = np.array(np.split(features, len(features) / group))
    labels = np.array(np.split(labels, len(labels) / group))
    
    # shuffle the order
    # DEBUG remove shuffling for now
    print('shuffling ..')
    indices = np.arange(features.shape[0])
    # this modify in place
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    # split the data into a training set and a validation set
    print('splitting ..')
    num_validation_samples = int(0.1 * features.shape[0])
    x_train = features[:-num_validation_samples*2]
    y_train = labels[:-num_validation_samples*2]
    x_val = features[-num_validation_samples*2:-num_validation_samples]
    y_val = labels[-num_validation_samples*2:-num_validation_samples]
    x_test = features[-num_validation_samples:]
    y_test = labels[-num_validation_samples:]

    # concate
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def prepare_data_using_tokenizer(articles, summaries, labels,
                                 tokenizer, group):
    """
    (?, 640). Each word is projected to its INDEX (int) in the tokenizer.
    """
    print('article texts to sequences ..')
    # TODO this is pretty slow. I parsed each article 21 times. I can
    # surely reduce this
    article_sequences = tokenizer.texts_to_sequences(articles)
    print('padding ..')
    article_sequences_padded = pad_sequences(article_sequences,
                                             value=0, padding='post',
                                             maxlen=MAX_ARTICLE_LENGTH)
    print('summary texts to sequences ..')
    summary_sequences = tokenizer.texts_to_sequences(summaries)
    print('padding ..')
    summary_sequences_padded = pad_sequences(summary_sequences,
                                             value=0, padding='post',
                                             maxlen=MAX_SUMMARY_LENGTH)
    print('concatenating ..')
    data = np.concatenate((article_sequences_padded,
                           summary_sequences_padded), axis=1)

    return shuffle_and_split(data, np.array(labels), group=group)

def prepare_summary_data_using_tokenizer(summaries, scores, tokenizer, group):
    """
    (?, 128)
    """
    print('summary texts to sequences ..')
    summary_sequences = tokenizer.texts_to_sequences(summaries)
    print('padding ..')
    summary_sequences_padded = pad_sequences(summary_sequences,
                                             value=0, padding='post',
                                             maxlen=MAX_SUMMARY_LENGTH)
    return shuffle_and_split(summary_sequences_padded, np.array(scores), group=group)

def sent_embed_articles(articles, maxlen, use_embedder, batch_size=10000):
    """
    BATCH_SIZE is how many articles to send to the embedder.
    
    Input: list of articles or summaries.

    1. break an article into sentences
    2. sentence encoding sentences into 512-dim vectors
    3. max sentence
    """
    # DEBUG remove this assignments
    # maxlen=10
    # batch_size = 10000
    
    sents = [sentence_split(a) for a in articles]
    sents_padded = pad_sequences(sents, value='', padding='post',
                                 maxlen=maxlen, dtype=object)
    shape = sents_padded.shape
    flattened = np.ndarray.flatten(sents_padded)
    
    splits = np.array_split(flattened, math.ceil(len(flattened) / batch_size))
    print('number of batch:', len(splits))
    ct = 0
    embedding_list = []
    # [use_embedder.embed(splits[0]) for _ in range(5)]
    # use_embedder.embed_session.close()
    
    for s in splits:
        ct += 1
        print('-- batch', ct)
        use_embedder.embed(s)
        # DEBUG: somehow memory is running out
        # embedding_list.append(use_embedder.embed(s))
    embedding = np.array(embedding_list)
    embedding_reshaped = np.reshape(embedding, shape + (embedding.shape[-1],))
    return embedding_reshaped

def split_sent_and_pad(articles, maxlen):
    res = []
    for article in articles:
        sents = sentence_split(article)
        sents_data = pad_sequences([sents], value='', padding='post',
                                   dtype=object, maxlen=maxlen)[0]
        # the shape is (#sent, 512)
        res.append(sents_data)
    return np.array(res)
    

def prepare_data_string(articles, summaries, scores):
    """
    Return a padded sequence of sentences.

    (#article, 10+3, string)
    """
    # (#article, 10)
    print('processing articles ..')
    article_data = split_sent_and_pad(articles, 10)
    article_data.shape
    # (#article, 3)
    print('processing summaries ..')
    summary_data = split_sent_and_pad(summaries, 3)
    summary_data.shape
    print('connecting ..')
    data = np.concatenate((article_data, summary_data), axis=1)
    # (#article, 13)
    data.shape

    return shuffle_and_split(data, np.array(scores))
def pad_shuffle_split_data(articles, summaries, labels,
                           article_pad_length, summary_pad_length,
                           group):
    dtype = np.array(articles[0]).dtype
    print('padding articles ..')
    articles = pad_sequences(articles, value=0, padding='post',
                             maxlen=article_pad_length, dtype=dtype)
    print('padding summaries ..')
    summaries = pad_sequences(summaries, value=0, padding='post',
                              maxlen=summary_pad_length, dtype=dtype)
    print('concatenating ..')
    data = np.concatenate((articles, summaries), axis=1)
    return shuffle_and_split(data, np.array(labels), group=group)

