#!/usr/bin/env python3

import os

import numpy as np

from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D
from keras.layers import Lambda, Reshape
from keras.layers import LSTM, Dropout, Masking, concatenate
from keras import regularizers
from keras.models import Model

import keras

from keras import backend as K

from antirouge.config import *

def build_model(embedding_method, label_type, embedding_layer,
                input_shape, architecture):
    embedded_input, sequence_input, x, preds = None, None, None, None
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
        # hidden_size = round(K.int_shape(embedded_input)[1] / 2)
        hidden_size = 128
        if embedding_method == 'glove':
            hidden_size = 25
        x = LSTM(hidden_size)(embedded_input)
        # x = keras.layers.GRU(hidden_size)(embedded_input)
        # x = Dropout(0.5)(x)
        # x = Dense(128, activation='relu')(x)
    elif architecture == 'FC':
        x = keras.layers.Flatten()(embedded_input)
        x = Dense(128, activation='relu')(x)
    elif architecture == '2-LSTM':
        lstm1, lstm2 = None, None
        if embedding_method == 'glove':
            article = Lambda(lambda x: x[:, :ARTICLE_MAX_WORD, :])(embedded_input)
            summary = Lambda(lambda x: x[:, ARTICLE_MAX_WORD:ARTICLE_MAX_WORD + SUMMARY_MAX_WORD, :])(embedded_input)
            lstm1 = LSTM(25)(Masking(mask_value=0)(article))
            lstm2 = LSTM(25)(Masking(mask_value=0)(summary))
        else:
            article = Lambda(lambda x: x[:, :ARTICLE_MAX_SENT, :])(embedded_input)
            summary = Lambda(lambda x: x[:, ARTICLE_MAX_SENT:ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, :])(embedded_input)
            lstm1 = LSTM(128)(Masking(mask_value=0.)(article))
            lstm2 = LSTM(128)(Masking(mask_value=0.)(summary))
        x = concatenate([lstm1, lstm2])
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

def build_glove_summary_only_model(embedding_layer):
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(SUMMARY_MAX_WORD,), dtype='int32')
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

