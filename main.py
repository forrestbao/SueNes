import tensorflow as tf
import numpy as np
import json
import random
import json
import re

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras.utils import plot_model

import scipy
import pickle

import os, sys

from model import load_embedding
from model import build_model
from utils import save_data, load_data
from utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer

from data import pad_shuffle_split_data
from data import prepare_data_using_USE
from data import prepare_data_with_USE, prepare_data_with_INFER
from data import prepare_data_using_tokenizer, prepare_summary_data_using_tokenizer
from data import load_word_mutated_data
from data import load_story_keys, create_tokenizer_by_key

from keras.preprocessing.sequence import pad_sequences

from config import *

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
        with open(os.path.join(d, 'story.pickle'), 'rb') as f:
            stories = pickle.load(f)
        neg_filename = {'neg': 'negative.pickle',
                        'mutate': 'word-mutated.pickle',
                        'shuffle': 'shuffle.pickle'}[fake_method]
        with open(os.path.join(d, neg_filename), 'rb') as f:
            negatives = pickle.load(f)
    return stories, negatives

def load_data_helper(fake_method, embedding_method, num_samples,
                     num_fake_samples, fake_extra_option=None):
    """return articles, reference_summaries, reference_labels,
    fake_summaries, fake_labels
    """
    tokenizer = None
    stories, negatives = load_pickles(fake_method, embedding_method)
    story_keys = set(stories.keys())
    negative_keys = set(negatives.keys())
    keys = story_keys.intersection(negative_keys)
    keys = set(random.sample(keys, num_samples))
    articles = np.array([stories[key]['article'] for key in keys])
    reference_summaries = np.array([stories[key]['summary'] for key in keys])
    if fake_method == 'neg' or fake_method == 'shuffle':
        fake_summaries = np.array([negatives[key] for key in keys])
        fake_summaries = fake_summaries[:,:num_fake_samples]
        reference_labels = np.ones_like(reference_summaries, dtype=int)
        fake_labels = np.zeros_like(fake_summaries, dtype=int)
    elif fake_method == 'mutate':
        # add, delete, replace
        section = fake_extra_option
        # This is protocol
        fake_summaries = np.array([negatives[key][section]['text'] for key in keys])
        fake_labels = np.array([negatives[key][section]['label'] for key in keys])
        reference_labels = np.ones_like(reference_summaries, dtype=float)
    return (articles, reference_summaries, reference_labels,
            fake_summaries, fake_labels, keys)

def glove_summary_only_main():
    """TODO Use only summary data for prediction. The expected results
    should be bad.
    """
    data = prepare_summary_data_using_tokenizer(summaries, labels,
                                                tokenizer, group=group)
    model = build_glove_summary_only_model(embedding_layer)
    return
    
def pearson_correlation_old(y_true, y_pred):
    pearson_correlation = scipy.stats.pearsonr(y_true, y_pred)
    # (Pearson’s correlation coefficient, 2-tailed p-value)
    # return K.mean(y_pred)
    return pearson_correlation[0]

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
    fake_methods = ['neg', 'shuffle', 'mutate']
    embedding_methods = ['glove', 'USE', 'USE-Large', 'InferSent']
    architectures = ['CNN', 'FC', 'LSTM']
    num_samples = 1000
    num_samples = 10000
    num_samples = 30000
    for fake_method in fake_methods:
        for embedding_method in embedding_methods:
            for num_fake_samples in [1, 3]:
                for arch in architectures:
                    run_exp(fake_method, embedding_method,
                            num_samples, num_fake_samples,
                            architecture)
    run_exp('neg', 'USE', 10000, 1, 'LSTM')
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

def run_exp(fake_method, embedding_method, num_samples,
            num_fake_samples, architecture, fake_extra_option=None):
    assert(fake_method in ['neg', 'shuffle', 'mutate'])
    assert(embedding_method in ['glove', 'USE', 'USE-Large', 'InferSent'])
    assert(architecture in ['CNN', 'FC', 'LSTM'])
    
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
        article_pad_length = MAX_ARTICLE_LENGTH
        summary_pad_length = MAX_SUMMARY_LENGTH
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
        embedding_layer = load_embedding(tokenizer)
    else:
        embedding_layer = None
    # input_sahpe
    if embedding_method == 'glove':
        input_shape = (MAX_ARTICLE_LENGTH + MAX_SUMMARY_LENGTH,)
    elif embedding_method == 'USE':
        input_shape = (ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 512)
    else:
        input_shape = (ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 4096)
    # build model
    model = build_model(embedding_method, label_type, embedding_layer,
                        input_shape, architecture)

    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    if label_type == 'regression':
        loss = 'mse'
        metrics = ['mae', 'accuracy', pearson_correlation_f]
    else:
        loss = 'binary_crossentropy'
        metrics=['accuracy', pearson_correlation_f]
    # optimizer=tf.train.AdamOptimizer(0.01)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print('training ..')
    history = model.fit(x_train, y_train, epochs=20, batch_size=128,
                        validation_data=(x_val, y_val), verbose=1)
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
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])
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
    




# if __name__ == '__main__':
#     use_vector_main()
