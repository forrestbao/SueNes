from antirouge.tf2 import *
import numpy as np
import json
import random
import json
import re
import math

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras.utils import plot_model
import keras

import scipy
import pickle

import os, sys

from antirouge.model import build_model
from antirouge.embedding import load_glove_layer
from antirouge.utils import save_data, load_data
from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from antirouge.utils import dict_pickle_read, dict_pickle_read_keys, dict_pickle_write

from antirouge.data import pad_shuffle_split_data
from antirouge.data import load_story_keys, create_tokenizer_by_key

from keras.preprocessing.sequence import pad_sequences

from antirouge.config import *

def load_article_and_summary_string(keys):
    """Return (articles, summaries)"""
    with open(STORY_PICKLE_FILE, 'rb') as f:
        stories = pickle.load(f)
        articles = [stories[key]['article'] for key in keys]
        summaries = [stories[key]['summary'] for key in keys]
        return np.array(articles), np.array(summaries)

def load_negative_summary_string(keys):
    with open(NEGATIVE_SAMPLING_FILE, 'rb') as f:
        neg = pickle.load(f)
        return np.array([neg[key] for key in keys])
    
        

def merge_summaries(articles, reference_summaries, reference_labels,
                    fake_summaries, fake_labels):
    """
    (A) => (A,1)
    (A,B)
    """
    assert(articles.shape == reference_summaries.shape)
    assert(articles.shape == reference_labels.shape)
    assert(fake_summaries.shape == fake_labels.shape)
    assert(articles.shape[0] == fake_summaries.shape[0])
    A = articles.shape[0]
    B = fake_summaries.shape[1]
    
    articles = np.repeat(articles, B+1)
    reference_summaries = np.reshape(reference_summaries, (A,1))
    summaries = np.concatenate((reference_summaries, fake_summaries), axis=1)
    reference_labels = np.reshape(reference_labels, (A, 1))
    labels = np.concatenate((reference_labels, fake_labels), axis=1)
    summaries = np.ndarray.flatten(summaries)
    labels = np.ndarray.flatten(labels)
    return articles, summaries, labels

def load_pickles(fake_method, embedding_method):
    """Load true stroies pickle and negative pickle.
    """
    if embedding_method == 'glove':
        with open(STORY_PICKLE_FILE, 'rb') as f:
            stories = pickle.load(f)
        neg_file = {'neg': NEGATIVE_SAMPLING_FILE,
                    'mutate': WORD_MUTATED_FILE,
                    'shuffle': NEGATIVE_SHUFFLE_FILE}[fake_method]
        with open(neg_file, 'rb') as f:
            negatives = pickle.load(f)
    else:
        embed2dir = {'USE': USE_DAN_DIR,
                     'USE-Large': USE_TRANSFORMER_DIR,
                     'InferSent': INFERSENT_DIR}
        d = embed2dir[embedding_method]
        stories = dict_pickle_read(os.path.join(d, 'story'))
        neg_filename = {'neg': 'negative',
                        'mutate': 'mutated',
                        'shuffle': 'shuffle'}[fake_method]
        negatives = dict_pickle_read(os.path.join(d, neg_filename))
    return stories, negatives

def test():
    fake_method = 'mutate'
    embedding_method = 'USE-Large'
    num_samples = 100
    num_fake_samples = 1
    fake_extra_option = 'add'

def load_data_helper(fake_method, embedding_method, num_samples,
                     num_fake_samples, fake_extra_option=None):
    """return articles, reference_summaries, reference_labels,
    fake_summaries, fake_labels
    """
    # embedding_method = 'USE'
    print('loading pickle ..')
    stories, negatives = load_pickles(fake_method, embedding_method)
    story_keys = set(stories.keys())
    negative_keys = set(negatives.keys())
    keys = story_keys.intersection(negative_keys)
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "rb") as f:
            saved_keys = pickle.load(f)
            assert(saved_keys.issubset(keys))
            keys = saved_keys
            print("Load saved keys.")
    else:
        keys = set(random.sample(keys, num_samples))
        with open(KEYS_FILE, "wb") as f:
            pickle.dump(keys, f)
            print("Saving sample keys.")
    print(("Number of sample: ", len(keys)))
    
    print('retrieving article ..')
    articles = np.array([stories[key]['article'] for key in keys])
    reference_summaries = np.array([stories[key]['summary'] for key in keys])
    
    if fake_method == 'neg' or fake_method == 'shuffle':
        fake_summaries = np.array([negatives[key] for key in keys])
        fake_summaries = fake_summaries[:,:num_fake_samples]
        reference_labels = np.ones_like(reference_summaries, dtype=int)
        # DEBUG
        fake_labels = np.zeros_like(fake_summaries, dtype=int)
        # fake_labels = np.ones_like(fake_summaries, dtype=int)
        # fake_labels = - fake_labels
    elif fake_method == 'mutate':
        # add, delete, replace
        section = fake_extra_option
        # section = 'delete'
        # HACK
        if embedding_method != 'glove':
            valid_pred = lambda k: negatives[k][section]['text'].shape == (10,)
            valid_keys = [key for key in keys if valid_pred(key)]
            if len(valid_keys) < len(keys):
                print('Warning: removed invalid samples. Valid:',
                      len(valid_keys), 'all:', len(keys))
            # HACK
            keys = valid_keys
            articles = np.array([stories[key]['article'] for key in keys])
            reference_summaries = np.array([stories[key]['summary'] for key in keys])
        # Resume normal
        # This is protocol
        fake_summaries = np.array([negatives[key][section]['text'] for
                                   key in keys])
        fake_summaries = fake_summaries[:,:num_fake_samples]
        fake_labels = np.array([negatives[key][section]['label'] for key in keys])
        fake_labels = fake_labels[:,:num_fake_samples]
        reference_labels = np.ones_like(reference_summaries, dtype=float)
    else:
        raise Exception()
    return (articles, reference_summaries, reference_labels,
            fake_summaries, fake_labels, keys)

def pearson_correlation_f(y_true, y_pred):
    #being K.mean a scalar here, it will be automatically subtracted
    #from all elements in y_pred
    fsp = y_pred - K.mean(y_pred)
    fst = y_true - K.mean(y_true)

    devP = K.std(y_pred)
    devT = K.std(y_true)
    return K.mean(fsp*fst)/(devP*devT)

def test():
    num_samples = 1000
    num_samples = 100
    num_fake_samples = 1
    fake_method = 'neg'
    fake_method = 'shuffle'
    fake_method = 'mutate'
    embedding_method = 'glove'
    embedding_method = 'USE'
    architecture = 'CNN'
    fake_extra_option = 'delete'

    
def main():
    # setting parameters
    fake_methods = ['neg', 'shuffle', 'mutate']
    embedding_methods = ['glove', 'USE', 'USE-Large', 'InferSent']
    architectures = ['CNN', 'FC', 'LSTM']
    mutate_extra_options = ['add', 'delete', 'replace']

    # random settings
    num_samples = 100
    num_samples = 1000
    num_samples = 10000
    num_samples = 30000
    fake_method = 'neg'
    fake_method = 'shuffle'
    fake_method = 'mutate'
    num_fake_samples = 1
    fake_extra_option = 'add'

    # Setting:  mutate USE-Large 5000 1 CNN add
    run_all_exp(['neg'], ['glove', 'USE', 'USE-Large', 'InferSent'],
                [100], [1], ['CNN', 'FC'])

    run_exp('neg', 'InferSent', 30000, 1, 'CNN')
    run_exp('neg', 'glove', 30000, 1, 'CNN')
    run_exp('neg', 'USE-Large', 30000, 1, 'CNN')

    # Negative sampling experiment
    run_all_exp(['neg'], ['glove', 'USE', 'USE-Large'],
                [5000], [1], ['CNN', 'FC', 'LSTM'])
    run_all_exp(['neg'], ['glove', 'USE', 'USE-Large'],
                [5000], [1], ['CNN'])
    # mutate experiment
    run_all_exp(['mutate'], ['glove', 'USE', 'USE-Large'],
                [30000], [1], ['CNN', 'FC', 'LSTM'],
                ['add', 'delete', 'replace'])
    run_all_exp(['neg'], ['USE', 'USE-Large'],
                [30000], [1], ['LSTM'])
    
    run_all_exp(['mutate'], ['glove', 'USE', 'USE-Large', 'InferSent'],
                [5000], [1], ['FC'], ['add', 'delete', 'replace'])
    run_all_exp(['mutate'], ['glove', 'USE', 'USE-Large', 'InferSent'],
                [5000], [1], ['LSTM'], ['add', 'delete', 'replace'])
    
    run_all_exp(['mutate'], [
        # 'glove', 'USE', 'USE-Large',
                             'InferSent'],
                [100], [1], ['CNN', 'FC'], ['add', 'delete', 'replace'])


    run_exp('mutate', 'USE', 100, 1, 'CNN', 'replace')
    run_exp('mutate', 'USE-Large', 100, 1, 'CNN', 'delete')
    
    run_exp('neg', 'InferSent', 100, 1, 'CNN')
    run_exp('mutate', 'InferSent', 10000, 1, 'CNN', 'add')

    run_exp('neg', 'USE', 10000, 1, 'LSTM')
    run_exp('mutate', 'glove', 10000, 1, 'LSTM', 'add')
    
    run_exp('neg', 'USE', 30000, 1, 'LSTM')
    run_exp('neg', 'USE', 10000, 1, 'FC')
    
    run_exp('neg', 'glove', 10000, 1, 'LSTM')
    run_exp('neg', 'glove', 1000, 1, 'LSTM')
    run_exp('neg', 'glove', 30000, 1, 'FC')
    
    run_exp('neg', 'glove', 10000, 1, 'CNN')
    run_exp('shuffle', 'USE', 1000, 1, 'CNN')
    
    run_exp('neg', 'USE', 10000, 1, 'CNN')
    
    run_exp('neg', 'glove', 10000, 1, 'CNN')
    run_exp('neg', 'USE', 1000, 1, 'CNN')
    run_exp('neg', 'USE', 10000, 1, 'LSTM')
    run_exp('neg', 'USE', 10000, 1, 'FC')
    run_exp('neg', 'USE-Large', 10000, 1, 'CNN')
    run_exp('neg', 'InferSent', 10000, 1, 'CNN')
    run_exp('neg', 'glove', 10000, 5, 'CNN')
    
    run_exp('mutate', 'glove', 100, 1, 'CNN', 'add')
    run_exp('mutate', 'glove', 100, 1, 'CNN', 'delete')
    run_exp('mutate', 'glove', 100, 1, 'CNN', 'replace')
    return

def run_all_exp(fake_methods, embedding_methods, num_samples_list,
                num_fake_samples_list, architectures,
                fake_extra_options=[]):
    for f in fake_methods:
        for e in embedding_methods:
            for ns in num_samples_list:
                for nf in num_fake_samples_list:
                    for a in architectures:
                        if fake_extra_options:
                            for option in fake_extra_options:
                                res = run_exp(f,e,ns,nf,a,option)
                                print('SETTING:', f,e,ns,nf,a,option)
                                print('RESULT:', res)
                        else:
                            res = run_exp(f,e,ns,nf,a)
                            print('SETTING:', f,e,ns,nf,a)
                            print('RESULT:', res)


def load_infersent_data(fake_method, fake_extra_option):
    # with open('InferSent-mutate-add/0.padded', 'rb') as f:
    #     (x_train, y_train), (x_val, y_val), (x_test, y_test) = pickle.load(f)
    assert(fake_method in ['neg', 'mutate'])
    assert(fake_extra_option in [None, 'add', 'delete', 'replace'])
    if fake_method == 'neg':
        folder = 'mnt/data/InferSent-neg-None'
    elif fake_method == 'mutate':
        folder = 'mnt/data/InferSent-mutate-' + fake_extra_option
    if not os.path.exists(folder):
        print(folder, 'does not exist.')
        raise Exception()
    data = []
    for i in range(30):
        print('loading', i, '..')
        with open(os.path.join(folder, str(i) + '.padded'), 'rb') as f:
            p = pickle.load(f)
            data.append(p)
    print('concatenating ..')
    x_train = np.concatenate([d[0][0] for d in data])
    y_train = np.concatenate([d[0][1] for d in data])
    x_val = np.concatenate([d[1][0] for d in data])
    y_val = np.concatenate([d[1][1] for d in data])
    x_test = np.concatenate([d[2][0] for d in data])
    y_test = np.concatenate([d[2][1] for d in data])
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def test():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_infersent_data('neg')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_infersent_data('mutate', 'add')

def test():
    create_data_pipeline_1('neg')
    
    create_data_pipeline_1('mutate', 'add')
    create_data_pipeline_1('mutate', 'delete')
    create_data_pipeline_1('mutate', 'replace')
    
    create_data_pipeline_2(0, 'neg')
    create_data_pipeline_2(1, 'neg')
    create_data_pipeline_2(11, 'neg')

    for i in range(30):
        create_data_pipeline_2(i, 'mutate', 'replace')
        # create_data_pipeline_2(i, 'neg')

    with open('InferSent-neg-None/0.raw', 'rb') as f:
        p = pickle.load(f)

def create_data_pipeline_1(fake_method, fake_extra_option=None):
    embedding_method = 'InferSent'
    num_samples = 30000
    num_fake_samples = 1
    split=1000
    folder = '-'.join([embedding_method, fake_method,
                       str(fake_extra_option)])
    folder = os.path.join('mnt/data', folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('loading data ..')
    (articles, reference_summaries, reference_labels, fake_summaries,
     fake_labels, keys) = load_data_helper(fake_method,
                                           embedding_method,
                                           num_samples,
                                           num_fake_samples,
                                           fake_extra_option)
    print('merging data ..')
    articles, summaries, labels = merge_summaries(articles,
                                                  reference_summaries,
                                                  reference_labels,
                                                  fake_summaries,
                                                  fake_labels)
    # split into separate files
    group = num_fake_samples + 1
    real_split = split * group
    print('real_split:', real_split)
    print('len(articles):', len(articles))
    num_splits = range(math.ceil(len(articles) / real_split))
    print('num_splits:', num_splits)
    for i in num_splits:
        print('-- split', i, '..')
        start = i * real_split
        end = (i + 1) * real_split
        outfile = os.path.join(folder, str(i) + '.raw')
        if os.path.exists(outfile):
            continue
        res = (articles[start:end], summaries[start:end],
               labels[start:end])
        with open(outfile, 'wb') as f:
            pickle.dump(res, f, protocol=4)
            
def create_data_pipeline_2(part_num, fake_method, fake_extra_option=None):
    embedding_method = 'InferSent'
    article_pad_length = ARTICLE_MAX_SENT
    summary_pad_length = SUMMARY_MAX_SENT
    folder = '-'.join([embedding_method, fake_method,
                       str(fake_extra_option)])
    folder = os.path.join('mnt/data', folder)
    # for each file
    group = 2
    infile = os.path.join(folder, str(part_num) + '.raw')
    outfile = os.path.join(folder, str(part_num) + '.padded')
    if os.path.exists(outfile):
        print('already exists')
        return
    with open(infile, 'rb') as f:
        articles, summaries, labels = pickle.load(f)
    data = pad_shuffle_split_data(articles, summaries, labels,
                                  article_pad_length,
                                  summary_pad_length, group)
    # save the data
    print('writing to', outfile, '..')
    with open(outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

def test():
    run_infersent('neg', None, 'FC')
    run_infersent('neg', None, 'CNN')
    run_infersent('neg', None, 'LSTM')
    run_infersent('mutate', 'add', 'FC')
    run_infersent('mutate', 'add', 'CNN')
    run_infersent('mutate', 'add', 'LSTM')


def run_infersent(fake_method, fake_extra_option, architecture):
    # (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_infersent_data('neg')
    # (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_infersent_data('mutate', 'add')
    print('loading data ..')
    ((x_train, y_train),
     (x_val, y_val),
     (x_test, y_test)) = load_infersent_data(fake_method, fake_extra_option)
    print('train: ', x_train.shape, y_train.shape)
    print('val: ', x_val.shape, y_val.shape)
    print('test: ', x_test.shape, y_test.shape)

    print('building model ..')
    # build model
    label_dict = {'neg': 'classification',
                  'mutate': 'regression'}
    label_type = label_dict[fake_method]
    # embedding_layer
    input_shape = (ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 4096)
    # build model
    print('building model ..')
    model = build_model('InferSent', label_type, None, input_shape,
                        architecture)
    model.summary()
    if label_type == 'regression':
        loss = 'mse'
        metrics = ['mae', 'mse', 'accuracy', pearson_correlation_f]
    elif label_type == 'classification':
        loss = 'binary_crossentropy'
        metrics=['accuracy', pearson_correlation_f]
    else:
        raise Exception()
    num_epochs = 60
    # optimizer=tf.train.AdamOptimizer(0.01)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer = keras.optimizers.SGD(lr=0.00001, clipnorm=1.)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print('training ..')
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=128,
                        validation_data=(x_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                 min_delta=0,
                                                                 patience=3,
                                                                 verbose=0,
                                                                 mode='auto')],
                        verbose=1)
    result = model.evaluate(x_test, y_test)
    print('Test result: ', result)
    return result[1]
    

def run_exp(fake_method, embedding_method, num_samples,
            num_fake_samples, architecture, fake_extra_option=None):
    assert(fake_method in ['neg', 'shuffle', 'mutate'])
    assert(embedding_method in ['glove', 'USE', 'USE-Large', 'InferSent'])
    assert(architecture in ['CNN', 'FC', 'LSTM', '2-LSTM'])

    print('Setting: ', fake_method, embedding_method, num_samples,
          num_fake_samples, architecture, fake_extra_option)
    
    print('loading data ..')
    (articles, reference_summaries, reference_labels, fake_summaries,
     fake_labels, keys) = load_data_helper(fake_method,
                                           embedding_method,
                                           num_samples,
                                           num_fake_samples,
                                           fake_extra_option)
    print('merging data ..')
    articles, summaries, labels = merge_summaries(articles,
                                                  reference_summaries,
                                                  reference_labels,
                                                  fake_summaries,
                                                  fake_labels)
    if embedding_method == 'glove':
        # convert from string to sequence
        print('creating tokenizer ..')
        tokenizer = create_tokenizer_by_key(keys)
        articles = tokenizer.texts_to_sequences(articles)
        summaries = tokenizer.texts_to_sequences(summaries)
    if embedding_method == 'glove':
        article_pad_length = ARTICLE_MAX_WORD
        summary_pad_length = SUMMARY_MAX_WORD
    else:
        article_pad_length = ARTICLE_MAX_SENT
        summary_pad_length = SUMMARY_MAX_SENT
    print('padding ..')
    group = num_fake_samples + 1
    data = pad_shuffle_split_data(articles, summaries, labels,
                                  article_pad_length,
                                  summary_pad_length, group)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    
    print('train: ', x_train.shape, y_train.shape)
    print('val: ', x_val.shape, y_val.shape)
    print('test: ', x_test.shape, y_test.shape)

    print('building model ..')
    # build model
    label_dict = {'neg': 'classification',
                  'shuffle': 'classification',
                  'mutate': 'regression'}
    label_type = label_dict[fake_method]
    # embedding_layer
    if embedding_method == 'glove':
        embedding_layer = load_glove_layer(tokenizer.word_index)
    else:
        embedding_layer = None
    # input_sahpe
    if embedding_method == 'glove':
        input_shape = (ARTICLE_MAX_WORD + SUMMARY_MAX_WORD,)
    elif embedding_method == 'USE' or embedding_method == 'USE-Large':
        input_shape = (ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 512)
    else:
        input_shape = (ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 4096)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    # build model
    model = build_model(embedding_method, label_type, embedding_layer,
                        input_shape, architecture)

    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    if label_type == 'regression':
        loss = 'mse'
        metrics = ['mae', 'mse', 'accuracy', pearson_correlation_f]
    elif label_type == 'classification':
        loss = 'binary_crossentropy'
        # loss = 'hinge'
        # loss = 'categorical_hinge'
        metrics=['accuracy', pearson_correlation_f]
    else:
        raise Exception()
    num_epochs = 60
    # optimizer=tf.train.AdamOptimizer(0.01)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer = keras.optimizers.SGD(lr=0.00001, clipnorm=1.)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print('training ..')
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=128,
                        validation_data=(x_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                 min_delta=0,
                                                                 patience=3,
                                                                 verbose=0,
                                                                 mode='auto')],
                        verbose=1)
    # plot history
    filename = '-'.join([fake_method, embedding_method,
                         str(num_samples), str(num_fake_samples),
                         architecture]) + '.png'
    plot_history(history, filename)
    # print out test results
    result = model.evaluate(x_test, y_test)
    print('Test result: ', result)
    # return history, result
    # accuracy
    return result[1]

def run_exp2(fake_method, embedding_method, num_samples,
            num_fake_samples, architecture, fake_extra_option=None):
    assert(fake_method in ['neg', 'shuffle', 'mutate'])
    assert(embedding_method in ['glove', 'USE', 'USE-Large', 'InferSent'])
    assert(architecture in ['CNN', 'FC', 'LSTM', '2-LSTM'])

    print('Setting: ', fake_method, embedding_method, num_samples,
          num_fake_samples, architecture, fake_extra_option)
    
    print('loading data ..')
    (articles, reference_summaries, reference_labels, fake_summaries,
     fake_labels, keys) = load_data_helper(fake_method,
                                           embedding_method,
                                           num_samples,
                                           num_fake_samples,
                                           fake_extra_option)
    print('merging data ..')
    articles, summaries, labels = merge_summaries(articles,
                                                  reference_summaries,
                                                  reference_labels,
                                                  fake_summaries,
                                                  fake_labels)
    if embedding_method == 'glove':
        # convert from string to sequence
        print('creating tokenizer ..')
        tokenizer = create_tokenizer_by_key(keys)
        articles = np.array(tokenizer.texts_to_sequences(articles))
        summaries = np.array(tokenizer.texts_to_sequences(summaries))
    if embedding_method == 'glove':
        article_pad_length = ARTICLE_MAX_WORD
        summary_pad_length = SUMMARY_MAX_WORD
    else:
        article_pad_length = ARTICLE_MAX_SENT
        summary_pad_length = SUMMARY_MAX_SENT
    print('padding ..')
    group = num_fake_samples + 1


    articles = np.array(np.split(articles, len(articles) / group))
    summaries = np.array(np.split(summaries, len(summaries) / group))
    labels = np.array(np.split(labels, len(labels) / group))

    indices = None
    if os.path.exists(SHUFFLE_FILE):
        with open(SHUFFLE_FILE, "rb") as f:
            indices = pickle.load(f)
            assert(indices.shape[0] == articles.shape[0])
            print("Load saved shuffled index.")
    else:
        indices = np.arange(articles.shape[0])
        np.random.shuffle(indices)
        with open(SHUFFLE_FILE, "wb") as f:
            pickle.dump(indices, f)
            print("Saving shuffled index.")
    
    num_validation_samples = int(0.1 * articles.shape[0])
    train = indices[:-num_validation_samples*2]
    val = indices[-num_validation_samples*2:-num_validation_samples]
    test = indices[-num_validation_samples:]

    def data_generator(shuffle, batch_size):
        # batch counts group
        batch_size = int(math.ceil(batch_size / group))
        i = 0
        dtype = 'int32' if type(articles[0][0]) == ([]) else 'float32'
        while 1:
            j = 0
            x, y = [], []
            # '''
            while i + j < shuffle.shape[0] and j < batch_size:
                index = shuffle[i+j]
                article = pad_sequences(articles[index], value=0, padding='post',
                             maxlen=article_pad_length, dtype=dtype)
                summary = pad_sequences(summaries[index], value=0, padding='post',
                              maxlen=summary_pad_length, dtype=dtype)
                x.append(np.concatenate((article, summary), axis=1))
                y.append(labels[index])
                j += 1
            # '''
            '''
            article = np.concatenate(articles[shuffle[i:max(shuffle.shape[0], i+batch_size)]])
            summary = np.concatenate(summaries[shuffle[i:max(shuffle.shape[0], i+batch_size)]])
            label = np.concatenate(labels[shuffle[i:max(shuffle.shape[0], i+batch_size)]])
            x1 = pad_sequences(article, value=0, padding='post',
                             maxlen=article_pad_length, dtype=dtype)
            x2 = pad_sequences(summary, value=0, padding='post',
                             maxlen=summary_pad_length, dtype=dtype)
            x = np.concatenate((x1, x2), axis=1)
            '''
            x = np.concatenate(x)
            y = np.concatenate(y)
            yield (x, y)
            # yield (x, label)
            i += batch_size
            if i >= shuffle.shape[0]:
                i = 0
                np.random.shuffle(shuffle)
            
    '''
    data = pad_shuffle_split_data(articles, summaries, labels,
                                  article_pad_length,
                                  summary_pad_length, group)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    '''
    print('train: ', train.shape)
    print('val: ', val.shape)
    print('test: ', test.shape)
    
    print('building model ..')
    # build model
    label_dict = {'neg': 'classification',
                  'shuffle': 'classification',
                  'mutate': 'regression'}
    label_type = label_dict[fake_method]
    # embedding_layer
    if embedding_method == 'glove':
        embedding_layer = load_glove_layer(tokenizer.word_index)
    else:
        embedding_layer = None
    # input_sahpe
    if embedding_method == 'glove':
        input_shape = (ARTICLE_MAX_WORD + SUMMARY_MAX_WORD,)
    elif embedding_method == 'USE' or embedding_method == 'USE-Large':
        input_shape = (ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 512)
    else:
        input_shape = (ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 4096)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    # build model
    model = build_model(embedding_method, label_type, embedding_layer,
                        input_shape, architecture)

    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    if label_type == 'regression':
        loss = 'mse'
        metrics = [pearson_correlation_f, 'accuracy', 'mse', 'mae']
    elif label_type == 'classification':
        loss = 'binary_crossentropy'
        # loss = 'hinge'
        # loss = 'categorical_hinge'
        metrics=['accuracy', pearson_correlation_f]
    else:
        raise Exception()
    num_epochs = 60
    # optimizer=tf.train.AdamOptimizer(0.01)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer = keras.optimizers.SGD(lr=0.00001, clipnorm=1.)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print('training ..')
    '''
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=128,
                        validation_data=(x_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                 min_delta=0,
                                                                 patience=3,
                                                                 verbose=0,
                                                                 mode='auto')],
                        verbose=1)
    '''
    batch_size = 128
    history = model.fit_generator(data_generator(train, batch_size=batch_size), 
                    epochs=num_epochs,
                    steps_per_epoch = math.ceil(train.shape[0] / float(batch_size)),
                    validation_data=data_generator(val, batch_size=batch_size),
                    validation_steps=math.ceil(val.shape[0] / float(batch_size)),
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                min_delta=0,
                                                                patience=3,
                                                                verbose=0,
                                                                mode='auto')],
                    verbose=1)
    # plot history
    filename = '-'.join([fake_method, embedding_method,
                         str(num_samples), str(num_fake_samples),
                         architecture]) + '.png'
    plot_history(history, filename)
    # print out test results
    # result = model.evaluate(x_test, y_test)
    result = model.evaluate_generator(data_generator(test, batch_size=batch_size),
        steps=math.ceil(test.shape[0] / float(batch_size)))
    print('Test result: ', result)
    # return history, result
    # accuracy
    return result[1]

def plot_history(history, filename):
    """
    Plot acc and loss in the same figure.
    """
    # filename = 'history.png'
    # Plot training & validation accuracy values
    fig, axe = plt.subplots(nrows=1, ncols=2)
    # plt.figure()
    ax = axe[0]
    if 'acc' in history.history and 'val_acc' in history.history:
        ax.plot(history.history['acc'])
        ax.plot(history.history['val_acc'])
    else:
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    
    # Plot training & validation loss values
    # file2 = 'history2.png'
    # plt.figure()
    ax = axe[1]
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig(file2)
    fig.savefig(filename)
    # plt.show()
    # plt.savefig(filename)


def test():
    # Negative sampling experiment
    run_all_exp(['neg'], ['glove', 'USE', 'USE-Large'],
                [30000], [1], ['CNN', 'FC', 'LSTM'])
    # mutate experiment
    run_all_exp(['mutate'], ['glove', 'USE', 'USE-Large'],
                [30000], [1], ['CNN', 'FC', 'LSTM'],
                ['add', 'delete', 'replace'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('fake', choices=['neg', 'mutate'])
    parser.add_argument('embedding', choices=['glove', 'USE',
                                              'USE-Large', 'InferSent'])
    parser.add_argument('arch', choices=['CNN', 'FC', 'LSTM'])
    parser.add_argument('--extra', choices=['add', 'delete',
                                            'replace'])

    args = parser.parse_args()

    if args.embedding == 'InferSent':
        run_infersent(args.fake, args.extra, args.arch)
    else:
        run_exp(args.fake, args.embedding, 30000, 1, args.arch, args.extra)

    # run_infersent('neg', None, 'LSTM')
    # run_infersent('mutate', 'add', 'LSTM')
    # python3 main.py neg InferSent CNN --extra=replace
    
    # python3 main.py mutate USE CNN --extra=replace
    # python3 main.py neg USE-Large LSTM

