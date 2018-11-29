import tensorflow as tf
import numpy as np
import json
import random
import json
import re

import matplotlib.pyplot as plt

# why keras.preprocessing is not exporting tokenizer_from_json ??
from keras_preprocessing.text import tokenizer_from_json

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras import layers

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

import os, sys

from model import load_embedding
from data import load_text_data, save_data, load_data
from data import prepare_data_using_use

# this seems to be not useful at all
MAX_NUM_WORDS = 200000

MAX_ARTICLE_LENGTH = 512
MAX_SUMMARY_LENGTH = 128

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
    articles, summaries, scores = load_text_data(size='medium')
    data = prepare_data_using_use(articles, summaries, scores)
    (x_train, y_train), (x_val, y_val) = data
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    # save/load the data
    print('saving data to use-vector.pickle ..')
    save_data(data, 'use-vector.pickle')
    # data2 = load_data('use-vector.pickle')

def glove_main():
    # data v2
    articles, summaries, scores = load_text_data(size='tiny')
    # this is pretty time consuming, so save it
    tokenizer = prepare_tokenizer(articles + summaries)
    # alternatively, save and load. Note that you must ensure to fit
    # on the same text.
    save_tokenizer(tokenizer)
    tokenizer = load_tokenizer()
    # this is also slow
    ((x_train, y_train),
     (x_val, y_val)) = prepare_data_using_tokenizer(articles,
                                                    summaries, scores,
                                                    tokenizer)
    # save and load the data
    # save_data(x_train, y_train, x_val, y_val)
    # (x_train, y_train), (x_val, y_val) = load_data()
    
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    # model v2
    embedding_layer = load_embedding(tokenizer)
    model = build_glove_model(embedding_layer)
    train_model((x_train, y_train), (x_val, y_val))

def train_model(model, data):
    (x_train, y_train), (x_val, y_val) = data
    # training op
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer=tf.train.AdamOptimizer(0.01)
    model.compile(optimizer=optimizer,
                  # loss='binary_crossentropy',
                  loss='mse',
                  # metrics=['accuracy']
                  metrics=['mae']
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
