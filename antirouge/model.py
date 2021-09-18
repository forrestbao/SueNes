#!/usr/bin/env python3

import os

import numpy as np

from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D
from keras.layers import Lambda, Reshape
from keras.layers import LSTM, Dropout, Masking, concatenate
from keras import regularizers
from keras.models import Model
import tensorflow as tf

import keras

from keras import backend as K

from antirouge import config
from antirouge.embedding import load_glove_layer
from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer, load_tokenizer_CNN




# DEPRECATED
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
    # elif architecture == 'CNN-2D':
    #     # Alternative 2D CNN model
    #     reshape = Reshape((embedded_sequence.shape[0], 512, 1))(embedded_input)
    #     conv = Conv2D(128, (5, 512), activation='relu', padding='valid')(reshape)
    #     x = GlobalMaxPooling2D()(conv)
    #     x = Dense(128, activation='relu')(x)
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
            article = Lambda(lambda x: x[:, :config.ARTICLE_MAX_WORD, :])(embedded_input)
            summary = Lambda(lambda x: x[:, config.ARTICLE_MAX_WORD:config.ARTICLE_MAX_WORD + config.SUMMARY_MAX_WORD, :])(embedded_input)
            lstm1 = LSTM(25)(Masking(mask_value=0)(article))
            lstm2 = LSTM(25)(Masking(mask_value=0)(summary))
        else:
            article = Lambda(lambda x: x[:, :config.ARTICLE_MAX_SENT, :])(embedded_input)
            summary = Lambda(lambda x: x[:, config.ARTICLE_MAX_SENT:config.ARTICLE_MAX_SENT + config.SUMMARY_MAX_SENT, :])(embedded_input)
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


def create_FC_model(embedding_size):
    """Sentence embedding model. The size is 512 for USE and 4096 for
InferSent."""
    article_input = keras.layers.Input(shape=(config.ARTICLE_MAX_SENT, embedding_size),
                                       dtype='float32')
    summary_input = keras.layers.Input(shape=(config.SUMMARY_MAX_SENT, embedding_size),
                                       dtype='float32')
    x = keras.layers.concatenate([article_input, summary_input], axis=1)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[article_input, summary_input],
                               outputs=preds)
    return model


def create_LSTM_model(embedding_size):
    """Sentence embedding model. The size is 512 for USE and 4096 for
InferSent."""
    article_input = keras.layers.Input(shape=(config.ARTICLE_MAX_SENT, embedding_size),
                                       dtype='float32')
    summary_input = keras.layers.Input(shape=(config.SUMMARY_MAX_SENT, embedding_size),
                                       dtype='float32')
    x = keras.layers.concatenate([article_input, summary_input], axis=1)
    
    # article = Lambda(lambda x: x[:, :config.ARTICLE_MAX_SENT, :])(embedded_input)
    # summary = Lambda(lambda x: x[:, config.ARTICLE_MAX_SENT:config.ARTICLE_MAX_SENT + config.SUMMARY_MAX_SENT, :])(embedded_input)
    # lstm1 = LSTM(64)(Masking(mask_value=0.)(article_input))
    # lstm2 = LSTM(64)(Masking(mask_value=0.)(summary_input))
    # x = concatenate([lstm1, lstm2])
    x = LSTM(128)(x)
    x = Dense(128, activation='relu')(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[article_input, summary_input],
                               outputs=preds)
    return model


def create_CNN1D_model(embedding_size):
    """Sentence embedding model. The size is 512 for USE and 4096 for
InferSent."""
    article_input = keras.layers.Input(shape=(config.ARTICLE_MAX_SENT, embedding_size),
                                       dtype='float32')
    summary_input = keras.layers.Input(shape=(config.SUMMARY_MAX_SENT, embedding_size),
                                       dtype='float32')
    x = keras.layers.concatenate([article_input, summary_input], axis=1)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[article_input, summary_input],
                               outputs=preds)
    return model


def create_glove_model():
    """Word embedding model. This differs from sentence embedding models in that
    it has embedding layer builtin."""
    tokenizer = load_tokenizer_CNN()
    glove_layer = load_glove_layer(tokenizer.word_index)
    article_input = keras.layers.Input(shape=(config.ARTICLE_MAX_WORD,),
                                       dtype='int32')
    summary_input = keras.layers.Input(shape=(config.SUMMARY_MAX_WORD,),
                                       dtype='int32')
    x = keras.layers.concatenate([article_input, summary_input], axis=1)
    x = glove_layer(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[article_input, summary_input],
                               outputs=preds)
    return model



#####################
## Model training
#####################

def train_model(model, data_iters):
    training_seq, validation_seq, testing_seq = data_iters
    optimizer = tf.train.RMSPropOptimizer(0.001)
    loss = 'binary_crossentropy'
    metrics=['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=0,
                                       mode='auto')
    mc = keras.callbacks.ModelCheckpoint('best_model.ckpt',
                                         # 'best_model.h5',
                                         monitor='val_loss', mode='auto', verbose=0,
                                         # glove model cannot be saved
                                         # entirely due to memory
                                         # error
                                         save_weights_only=True,
                                         save_best_only=True)
    model.fit_generator(training_seq, epochs=100, shuffle=False,
                        validation_data=validation_seq,
                        # use_multiprocessing=True,
                        # max_queue_size=30,
                        # workers=4,
                        callbacks=[es, mc])
    # saved_model = load_model('best_model.h5')
    model.load_weights('best_model.ckpt')
    loss, acc = model.evaluate_generator(testing_seq)
    print('Testing loss %s, acc %s' % (loss, acc))
