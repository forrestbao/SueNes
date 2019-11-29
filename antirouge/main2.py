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
    
    # Repeat articles to generate (article, summary, label)
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
    keys = set(random.sample(keys, num_samples))
    
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

def shuffle_and_split(articles, summaries, labels, group):
    """Use interval to control how the split happens. For example
    group=21.

    1. group into groups
    2. shuffle groups, split groups
    3. flatten groups
    4. (optional) shuffle again

    """
    print('splitting by group ..')
    articles = np.array(np.split(articles, len(articles) / group))
    summaries = np.array(np.split(summaries, len(summaries) / group))
    labels = np.array(np.split(labels, len(labels) / group))
    
    # shuffle the order
    # DEBUG remove shuffling for now
    print('shuffling ..')
    indices = np.arange(articles.shape[0])
    # this modify in place
    np.random.shuffle(indices)
    articles = articles[indices]
    summaries = summaries[indices]
    labels = labels[indices]

    # split the data into a training set and a validation set
    print('splitting ..')
    num_validation_samples = int(0.1 * articles.shape[0])
    x_train = [np.concatenate(articles[:-num_validation_samples*2]), 
        np.concatenate(summaries[:-num_validation_samples*2])]
    y_train = np.concatenate(labels[:-num_validation_samples*2])
    
    x_val = [np.concatenate(articles[-num_validation_samples*2:-num_validation_samples]),
        np.concatenate(summaries[-num_validation_samples*2:-num_validation_samples])]
    y_val = np.concatenate(labels[-num_validation_samples*2:-num_validation_samples])
    
    x_test = [np.concatenate(articles[-num_validation_samples:]),
        np.concatenate(summaries[-num_validation_samples:])]
    y_test = np.concatenate(labels[-num_validation_samples:])

    '''
    for i in range(x_train[0].shape[0]):
        x_train[0][i] = tf.convert_to_tensor(x_train[0][i])
        x_train[1][i] = tf.convert_to_tensor(x_train[1][i])

    for i in range(x_val[0].shape[0]):
        x_val[0][i] = tf.convert_to_tensor(x_val[0][i])
        x_val[1][i] = tf.convert_to_tensor(x_val[1][i])
    
    for i in range(x_test[0].shape[0]):
        x_test[0][i] = tf.convert_to_tensor(x_test[0][i])
        x_test[1][i] = tf.convert_to_tensor(x_test[1][i])
    '''
    # concate
    '''
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    '''
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)    

def run_exp(fake_method, embedding_method, num_samples,
            num_fake_samples, architecture, fake_extra_option=None):
    assert(fake_method in ['neg', 'shuffle', 'mutate'])
    assert(embedding_method in ['glove', 'USE', 'USE-Large', 'InferSent'])
    assert(architecture in ['CNN', 'FC', 'LSTM'])

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
    '''
    if embedding_method == 'glove':
        article_pad_length = ARTICLE_MAX_WORD
        summary_pad_length = SUMMARY_MAX_WORD
    else:
        article_pad_length = ARTICLE_MAX_SENT
        summary_pad_length = SUMMARY_MAX_SENT
    print('padding ..')
    '''
    group = num_fake_samples + 1

    ## Todo: modify here
    data = shuffle_and_split(articles, summaries, labels, group)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    
    print('train: ', x_train[0].shape, x_train[1].shape, y_train.shape)
    print('val: ', x_val[0].shape, x_val[1].shape, y_val.shape)
    print('test: ', x_test[0].shape, x_test[1].shape, y_test.shape)

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
    # build model
    # model = build_model(embedding_method, label_type, embedding_layer,
    #                     input_shape, architecture)
    article_input = tf.keras.Input(shape=(None, 512), dtype='float32')
    summary_input = tf.keras.Input(shape=(None, 512), dtype='float32')
    x = tf.keras.layers.LSTM(128)(article_input)
    y = tf.keras.layers.LSTM(128)(summary_input)
    z = None
    if label_type == 'classification':
        z = tf.keras.layers.Dense(1, activation='sigmoid')(tf.keras.layers.concatenate([x, y]))
    elif label_type == 'regression':
        z = tf.keras.layers.Dense(1)(tf.keras.layers.concatenate([x, y]))
    model = tf.keras.Model([article_input, summary_input], z)


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
    def train_generator():
        while 1:
            steps = x_train[0].shape[0]
            for i in range(steps):
                yield ([np.array([x_train[0][i]]), np.array([x_train[1][i]])], np.array([y_train[i]]))
            pass
    
    def val_generator():
        while 1:
            steps = x_val[0].shape[0]
            for i in range(steps):
                yield ([np.array([x_val[0][i]]), np.array([x_val[1][i]])], np.array([y_val[i]]))
            pass

    history = model.fit_generator(train_generator(), epochs=num_epochs, steps_per_epoch=x_train[0].shape[0],
                        validation_data=val_generator(), validation_steps=x_val[0].shape[0],
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
        pass
        # run_exp(args.fake, args.embedding, 30000, 1, args.arch, args.extra)

    # run_infersent('neg', None, 'LSTM')
    # run_infersent('mutate', 'add', 'LSTM')
    # python3 main.py neg InferSent CNN --extra=replace
    
    # python3 main.py mutate USE CNN --extra=replace
    # python3 main.py neg USE-Large LSTM

