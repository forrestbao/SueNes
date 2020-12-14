#!/usr/bin/env python3
import pickle
import numpy as np
import os
import json
import re
import math
import random

from keras.preprocessing.sequence import pad_sequences

from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer

from antirouge.config import *

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

