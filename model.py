#!/usr/bin/env python3

import os

import numpy as np

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D
from keras.layers import Lambda, Reshape
from keras import regularizers
from keras.models import Model
from keras.initializers import Constant

import keras

from keras import backend as K

from config import *

GLOVE_DIR = '/home/hebi/github/reading/keras/examples/data/glove.6B'

def load_embedding(tokenizer):
    """1. read glove.6B.100d embedding matrix
    
    2. from tokenizer, get the number of words, use it (with a MAX
    value) as the dimension of embedding matrix.
    
    3. for all the words in the tokenizer, (as long as its index is
    less than MAX value), fill the embedding matrix with its glove
    value

    4. from the matrix, create a embedding layer by pass the matrix as
    embeddings_initializer. This layer is fixed by setting it not
    trainable.

    """
    print('Indexing word vectors.')
    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                # MAX_SEQUENCE_LENGTH = 1000
                                # input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer

def build_model():
    """The model contains:
    """
    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64,
                           # input_shape=(768,),
                           activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    # model.add(layers.Dense(10, activation='softmax'))
    # output the score
    model.add(layers.Dense(1, activation='sigmoid'))

    # x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = GlobalMaxPooling1D()(x)
    # x = Dense(128, activation='relu')(x)
    
    return model

def build_glove_model(embedding_layer):
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_ARTICLE_LENGTH + MAX_SUMMARY_LENGTH,),
                           dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # FIXME The literature to apply CNN to NLP is to use the filter size
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model

def build_glove_2dCONV_model(embedding_layer):
    sequence_length = MAX_ARTICLE_LENGTH + MAX_SUMMARY_LENGTH
    sequence_input = Input(shape=(sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedded_sequences)
    # conv_0 = Conv2D(100, (3, EMBEDDING_DIM), activation='relu',
    #                 kernel_regularizer=regularizers.l2(0.01))(reshape)
    # conv_1 = Conv2D(100, (4, EMBEDDING_DIM), activation='relu',
    #                 kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_0 = Conv2D(100, (3, 3), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    maxpool_0 = MaxPooling2D((sequence_length - 3 + 1, 1), strides=(1,1))(conv_0)
    # maxpool_1 = MaxPooling2D((sequence_length - 4 + 1, 1), strides=(1,1))(conv_1)
    maxpool_0
    maxpool_1
    # merged_tensor = keras.layers.Concatenate()([maxpool_0, maxpool_1])
    # merged_tensor
    flatten = keras.layers.Flatten()(maxpool_0)
    flatten
    x = Dense(64, activation='relu')(flatten)
    preds = Dense(1)(x)
    model = Model(sequence_input, preds)
    return model
    
    

def build_glove_LSTM_model(embedding_layer):
    """Bad performance.
    """
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_ARTICLE_LENGTH + MAX_SUMMARY_LENGTH,),
                           dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences
    # x= keras.layers.LSTM(EMBEDDING_DIM, return_sequences=True)(embedded_sequences)
    x= keras.layers.LSTM(EMBEDDING_DIM)(embedded_sequences)
    x = keras.layers.Dropout(0.5)(x)
    
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    # x = keras.layers.TimeDistributed(Dense(128, activation='relu'))(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model

def build_binary_glove_model(embedding_layer):
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_ARTICLE_LENGTH + MAX_SUMMARY_LENGTH,),
                           dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

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
    
    # x = GlobalMaxPooling1D()(embedded_article)
    # flattened_article = keras.layers.Flatten()(embedded_article)
    # flattened_summary = keras.layers.Flatten()(embedded_summary)
    # article_vec = Dense(512)(flattened_article)
    # summary_vec = Dense(512)(flattened_summary)
    # article_vec
    # summary_vec
    # dotproduct = keras.layers.Dot(1)([article_vec, summary_vec])
    # dotproduct
    # preds = Dense(1, activation='sigmoid')(dotproduct)
    # preds
    # sequence_input

    
    # preds = Dense(1, activation='sigmoid')(x)
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

def build_uae_model():
    sequence_input = Input(shape=(ARTICLE_MAX_SENT + SUMMARY_MAX_SENT, 512),
                           dtype='float32')
    x = Conv1D(128, 3, activation='relu')(sequence_input)
    x = MaxPooling1D(2)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model
    

def build_model_test():
    # apply a layer multiple times
    i = Input(shape=(13,100), dtype='float32')
    i.shape
    l = Dense(512, input_dim=(100))
    tf.concat([l(t) for t in tf.split(i, 13, 1)], 1)
    x = tf.reshape(i, shape=(100,13))
    #
    # original:
    # 100 -> 512
    i = Dense(512, input_dim=(100))(i)
    return

    
def build_use_model(use_embed):
    """Since USE can only used CPU, I'd better not getting it into the
Model layers.
    """
    sequence_input = Input(shape=(13,), dtype=tf.string)
    # (?, 13)
    sequence_input
    
    # Simply doing this is not working:
    # >>> embedded_sequences = use_embed(sequence_input)
    # So instead, split the tensors and concatenate afterwards
    in_sub_tensors = tf.split(sequence_input, 13, 1)
    in_sub_tensors
    # takes time to construct
    # (?) to (512)
    out_sub_tensors = [use_embed(tf.reshape(t, [-1]))
                       for t in in_sub_tensors]
    embedded_sequences = tf.concat([tf.reshape(t, (-1, 1, 512))
                                 for t in out_sub_tensors], axis=1)
    # (?, 13, 512)
    embedded_sequences

    # testing use_embed:
    # >>> holder = tf.placeholder(tf.string, shape=(None))
    # >>> holder.shape
    # >>> holder
    # >>> similarity_message_encodings = use_embed(holder)
    # (13, 512)

    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model
