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
from model import build_uae_model, build_glove_model
from data import load_text_data, save_data, load_data
from data import prepare_tokenizer, save_tokenizer, load_tokenizer
from data import prepare_data_using_use,
from data import prepare_data_using_tokenizer, prepare_summary_data_using_tokenizer

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

def glove_main():
    # data v2
    articles, summaries, scores = load_text_data(size='medium')
    # this is pretty time consuming, so save it
    tokenizer = prepare_tokenizer(articles + summaries)
    # alternatively, save and load. Note that you must ensure to fit
    # on the same text.
    # save_tokenizer(tokenizer)
    # tokenizer = load_tokenizer()
    #
    # this is also slow
    data = prepare_data_using_tokenizer(articles, summaries, scores,
                                        tokenizer)
    # save and load the data
    # save_data(data, 'glove-data-10000.pickle')
    # data = load_data('glove-data-10000.pickle')
    (x_train, y_train), (x_val, y_val) = data
    
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    # model v2
    embedding_layer = load_embedding(tokenizer)
    model = build_glove_model(embedding_layer)
    train_model(model, data)
    return

def glove_summary_main():
    """Use only summary data for prediction. The expected results should
    be bad.

    """
    _, summaries, scores = load_text_data(size='medium')
    # this is pretty time consuming, so save it
    tokenizer = prepare_tokenizer(summaries)
    # alternatively, save and load. Note that you must ensure to fit
    # on the same text.
    # save_tokenizer(tokenizer)
    # tokenizer = load_tokenizer()
    #
    # this is also slow
    
    data = prepare_summary_data_using_tokenizer(articles, summaries,
                                                scores, tokenizer)
    # save and load the data
    # save_data(data, 'glove-data-10000.pickle')
    # data = load_data('glove-data-10000.pickle')
    (x_train, y_train), (x_val, y_val) = data
    
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    # model v2
    embedding_layer = load_embedding(tokenizer)
    # FIXME the architecture needs adjustment
    model = build_glove_model(embedding_layer)
    train_model(model, data)
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
              epochs=40, batch_size=128,
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
    (x_train, y_train), (x_val, y_val) = prepare_data()
    model = build_model()
    train_model(model, (x_train, y_train), (x_val, y_val))

if __name__ == '__main__':
    use_vector_main()
