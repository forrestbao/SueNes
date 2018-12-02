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

import scipy

import os, sys

from model import load_embedding
from model import build_uae_model, build_glove_model, build_glove_summary_only_model
from model import build_binary_glove_model, build_separate_model
from model import build_glove_LSTM_model, build_glove_2dCONV_model
from utils import save_data, load_data
from utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer

from data import prepare_data_using_use
from data import prepare_data_using_tokenizer, prepare_summary_data_using_tokenizer
from data import load_negative_sampling_data, load_article_and_summary_data
from data import load_word_mutated_data
from data import load_story_keys, create_tokenizer_by_key

from config import *

def use_string_main():
    articles, summaries, scores = load_text_data()
    (x_train, y_train), (x_val, y_val) = prepare_data_string(articles,
                                                             summaries,
                                                             scores)

    use_embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    model = build_use_model(use_embed)
    use_embed(['hello'])
    # training op
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer=tf.train.AdamOptimizer(0.01)
    model.compile(optimizer=optimizer,
                  # loss='binary_crossentropy',
                  loss='mse',
                  # metrics=['accuracy']
                  # TODO use correlation coefficient
                  metrics=['mae'])
    model.fit(x_train, y_train,
              epochs=40, batch_size=128,
              validation_data=(x_val, y_val), verbose=1)
    model.summary()
    return

def use_vector_main():
    """Use USE to get embedding, then feed into the network.
    """
    # articles, summaries, scores = load_text_data(size='medium')
    # data = prepare_data_using_use(articles, summaries, scores)
    data = prepare_data_using_use()
    (x_train, y_train), (x_val, y_val) = data
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    # save/load the data
    # print('saving data to use-vector.pickle ..')
    # save_data(data, 'use-vector.pickle')
    # data2 = load_data('use-vector.pickle')
    model = build_uae_model()
    train_model(model, data)
    return


def glove_neg_main():
    keys = load_story_keys(10000)
    # tokenizer = load_tokenizer()
    tokenizer = create_tokenizer_by_key(keys)
    fake_summaries = load_negative_sampling_data(keys)
    # DEBUG using one fake summary, to get unbiased data
    # fake_summaries = fake_summaries[:,:1]
    fake_summaries.shape
    articles, reference_summaries = load_article_and_summary_data(keys)
    reference_labels = np.ones_like(reference_summaries, dtype=int)
    fake_labels = np.zeros_like(fake_summaries, dtype=int)
    
    articles.shape
    reference_summaries.shape
    fake_summaries.shape
    fake_labels.shape

    res = concatenate_data(articles, reference_summaries,
                           reference_labels,
                           fake_summaries,
                           fake_labels)
    articles, summaries, labels = res
    articles.shape
    summaries.shape
    labels.shape

    group = fake_summaries.shape[1] + 1
    data = prepare_data_using_tokenizer(articles, summaries,
                                        labels, tokenizer, group=group)
    (x_train, y_train), (x_val, y_val) = data
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    embedding_layer = load_embedding(tokenizer)
    model = build_binary_glove_model(embedding_layer)
    model = build_separate_model(embedding_layer)
    train_binary_model(model, data)
    return

def concatenate_data(articles, reference_summaries, reference_labels,
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

def glove_main():
    # data v2
    keys = load_story_keys(1000)
    tokenizer = create_tokenizer_by_key(keys)
    articles, reference_summaries = load_article_and_summary_data(keys)
    fake_summaries, fake_labels = load_word_mutated_data(keys, mode='del')
    reference_labels = np.ones_like(reference_summaries, dtype=float)
    
    articles.shape
    reference_summaries.shape
    fake_summaries.shape
    fake_labels.shape

    res = concatenate_data(articles, reference_summaries,
                           reference_labels,
                           fake_summaries,
                           fake_labels)
    articles, summaries, labels = res
    articles.shape
    summaries.shape
    labels.shape
    
    group = fake_summaries.shape[1] + 1
    data = prepare_data_using_tokenizer(articles, summaries,
                                        labels, tokenizer, group=group)
    # summary only data
    data = prepare_summary_data_using_tokenizer(summaries, labels,
                                                tokenizer, group=group)
    # save and load the data
    (x_train, y_train), (x_val, y_val) = data
    
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    # model v2
    embedding_layer = load_embedding(tokenizer)

    model = build_glove_model(embedding_layer)
    model = build_separate_model(embedding_layer)
    model = build_glove_LSTM_model(embedding_layer)
    model = build_glove_summary_only_model(embedding_layer)
    model = build_glove_2dCONV_model(embedding_layer)
    model.summary()
    
    train_model(model, data)
    train_model_with_test(model, data)
    return

def glove_summary_main():
    """Use only summary data for prediction. The expected results should
    be bad.

    """
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

def train_binary_model(model, data):
    (x_train, y_train), (x_val, y_val) = data
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', pearson_correlation_f])
    model.fit(x_train, y_train,
              epochs=20, batch_size=128,
              validation_data=(x_val, y_val), verbose=1)
    model.summary()
    
def train_model_with_test(model, data):
    """(x_val, y_val) is used as test data. When fitting the model,
    validation_split is set to 0.2

    Seems that this is not very necessary.

    """
    (x_train, y_train), (x_val, y_val) = data
    # training op
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer=tf.train.AdamOptimizer(0.01)
    model.compile(optimizer=optimizer,
                  # loss='binary_crossentropy',
                  loss='mse',
                  # metrics=['accuracy']
                  metrics=['mae',
                           # tf.contrib.metrics.streaming_pearson_correlation,
                           pearson_correlation_f]
    )
    model.fit(x_train, y_train, epochs=20, batch_size=128,
              validation_split = 0.2, verbose=1)
    model.summary()
    eval_loss, eval_mae, eval_pearson = model.evaluate(x_val, y_val)
    print('loss:', eval_loss, 'mae:', eval_mae, 'pearson:', eval_pearson)

def train_model(model, data):
    (x_train, y_train), (x_val, y_val) = data
    # training op
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer=tf.train.AdamOptimizer(0.01)
    model.compile(optimizer=optimizer,
                  # loss='binary_crossentropy',
                  loss='mse',
                  # metrics=['accuracy']
                  metrics=['mae',
                           # tf.contrib.metrics.streaming_pearson_correlation,
                           pearson_correlation_f]
    )
    model.fit(x_train, y_train,
              epochs=20, batch_size=128,
              validation_data=(x_val, y_val), verbose=1)
    model.summary()
    # results = model.evaluate(x_test, y_test)
    
def main():
    """Steps:
    1. preprocess data:
      - tokenization (sentence tokenizer)
      - separate article and reference summary
      - chunk into train and test

    2. data generation: for each reference summary, do the following
    mutation operations: deletion, insertion, mutation. According to
    how much are changed, assign a score.
    
    3. sentence embedding: embed article and summary into sentence
    vectors. This is the first layer, the embedding layer. Then, do a
    padding to get the vector to the same and fixed dimension
    (e.g. summary 20, article 100). FIXME what to do for very long
    article? Then, fully connected layer directly to the final result.

    """
    print('Deprecated. Exiting ..')
    return 0
    (x_train, y_train), (x_val, y_val) = prepare_data()
    model = build_model()
    train_model(model, (x_train, y_train), (x_val, y_val))

# if __name__ == '__main__':
#     use_vector_main()
