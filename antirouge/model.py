#!/usr/bin/env python3

import os

import numpy as np

from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D
from keras.layers import Lambda, Reshape
from keras.layers import LSTM, Dropout
from keras import regularizers
from keras.models import Model

import keras

from keras import backend as K

from antirouge.config import *

def build_model(embedding_method, label_type, embedding_layer,
                input_shape, architecture):
    # Embedding layer
    if embedding_method == 'glove':
        sequence_input = Input(shape=input_shape, dtype='int32')
        # (640, 100)
        embedded_input = embedding_layer(sequence_input)
    else:
        # (13, 512) or (13, 4096)
        # input_shape = (13, 512)
        sequence_input = Input(shape=input_shape, dtype='float32')
        embedded_input = sequence_input
        
    # Architecture layer
    if architecture == 'CNN':
        # 1 layer CNN
        x = Conv1D(128, 5, activation='relu')(embedded_input)
        # x = MaxPooling1D(3)(x)
        x = GlobalMaxPooling1D()(x)
        # x = Dense(128, activation='relu')(x)
    elif architecture == 'CNN-3':
        # 3 layer CNN
        x = Conv1D(128, 5, activation='relu')(embedded_input)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
    elif architecture == 'CNN-2D':
        # Alternative 2D CNN model
        reshape = Reshape((embedded_sequence.shape[0], 512, 1))(embedded_input)
        conv = Conv2D(128, (5, 512), activation='relu', padding='valid')(reshape)
        x = GlobalMaxPooling2D()(conv)
        x = Dense(128, activation='relu')(x)
    elif architecture == 'LSTM':
        # DEBUG
        hidden_size = round(K.int_shape(embedded_input)[1] / 2)
        if hidden_size > 128:
            hidden_size = 128
        # hidden_size = 128
        x= LSTM(hidden_size)(embedded_input)
        # x = keras.layers.GRU(hidden_size)(embedded_input)
        # x = Dropout(0.5)(x)
        # x = Dense(128, activation='relu')(x)
    elif architecture == 'FC':
        x = keras.layers.Flatten()(embedded_input)
        x = Dense(128, activation='relu')(x)
    else:
        raise Exception()

    # Output layer
    if label_type == 'classification':
        preds = Dense(1, activation='sigmoid')(x)
    elif label_type == 'regression':
        preds = Dense(1)(x)
    else:
        raise Exception()

    model = Model(sequence_input, preds)
    return model

def build_separate_model(embedding_layer):
    sequence_input = Input(shape=(MAX_ARTICLE_LENGTH + MAX_SUMMARY_LENGTH,),
                           dtype='int32')
    # article_input = Input(shape=(MAX_ARTICLE_LENGTH,), dtype='int32')
    # summary_input = Input(shape=(MAX_SUMMARY_LENGTH,), dtype='int32')
    sequence_input
    # This is not layer
    # article_input = sequence_input[:, :MAX_ARTICLE_LENGTH]
    # summary_input = sequence_input[:, MAX_ARTICLE_LENGTH:]
    article_input = Lambda(lambda x: x[:,:MAX_ARTICLE_LENGTH])(sequence_input)
    summary_input = Lambda(lambda x: x[:,MAX_ARTICLE_LENGTH:])(sequence_input)
    article_input
    summary_input
    embedded_article = embedding_layer(article_input)
    embedded_summary = embedding_layer(summary_input)
    embedded_article
    embedded_summary

    # TODO add CNN to speed up LSTM training

    # x1 = keras.layers.LSTM(EMBEDDING_DIM, return_sequences=True)(embedded_article)
    x1 = keras.layers.LSTM(EMBEDDING_DIM)(embedded_article)
    # x2 = keras.layers.LSTM(EMBEDDING_DIM, return_sequences=True)(embedded_summary)
    x2 = keras.layers.LSTM(EMBEDDING_DIM)(embedded_summary)

    x1
    x2

    x = keras.layers.Concatenate()([x1, x2])
    x
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)
    
    model = Model(sequence_input, preds)
    return model

def build_glove_summary_only_model(embedding_layer):
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SUMMARY_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model

def __test_nce_loss():
    """Create a model to test NCE loss.

    1. is the labels always 1?

    The plan: y = 1 if 0<x<100 else 0
    
    """
    # generate data
    # how about using mnist data?
    
    pass

