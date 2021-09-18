#!/usr/bin/env python3

import sys
import os

# so that the tfhub module is not deleted after reboot
os.environ["TFHUB_CACHE_DIR"] = os.path.expanduser('~/.cache/tfhub_modules')

from antirouge.tf2 import *
#import tensorflow as tf
#import tensorflow_hub as hub
import torch
import numpy as np
import zipfile
import time
import os

import keras

from keras.layers import Embedding
from keras.initializers import Constant
from keras.utils.data_utils import get_file

from antirouge.utils import load_tokenizer

# pip install --user git+https://github.com/lihebi/InferSent
from infersent.models import InferSent

from antirouge import config

# http://nlp.stanford.edu/data/glove.6B.zip

# 50, 100, 200, 300
GLOVE_EMBEDDING_DIM = 100


def load_glove_matrix():
    # download glove embedding, unzip
    glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    path = get_file('glove.6B.zip', glove_url)
    # unzip
    fname = 'glove.6B.%sd.txt' % GLOVE_EMBEDDING_DIM
    glove_txt = os.path.join(os.path.dirname(path), fname)
    if not os.path.exists(glove_txt):
        zipfile.ZipFile(path).extractall(os.path.dirname(path))
    assert(os.path.exists(glove_txt))
    res = {}
    with open(glove_txt, encoding='UTF-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            res[word] = coefs
    return res
    
# adapted with keras/examples/pretrained_word_embeddings.py
def load_glove_layer(word_index):
    """word_index is a dictionary from word to its index (0-x). Can be
    tokenizer.word_index.

    1. read glove.6B.100d embedding matrix

    2. from tokenizer, get the number of words, use it (with a MAX
    value) as the dimension of embedding matrix.
    
    3. for all the words in the tokenizer, (as long as its index is
    less than MAX value), fill the embedding matrix with its glove
    value

    4. from the matrix, create a embedding layer by pass the matrix as
    embeddings_initializer. This layer is fixed by setting it not
    trainable.

    """
    glove_mat = load_glove_matrix()
    # prepare embedding matrix
    num_words = min(config.MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, GLOVE_EMBEDDING_DIM))
    for word, i in word_index.items():
        # FIXME ???
        # if i > config.NEGATIVE_SHUFFLE_FILEMAX_NUM_WORDS:
        #     continue
        embedding_vector = glove_mat.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                config.EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                # MAX_SEQUENCE_LENGTH = 1000
                                # input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer

tf.logging.set_verbosity(tf.logging.WARN)

# Some note about InferSent version:
#
# version 1 uses infersent1.pkl, glove.840B.300d.txt, and when
# creating InferSent model, need to pass 'version': 2 to it.
#
# version 2 uses infersent1.pkl and crawl-300d-2M.vec
def get_infersent_modelpath():
    return get_file('infersent2.pkl',
                    'https://dl.fbaipublicfiles.com/infersent/infersent2.pkl')
    
def get_infersent_w2vpath():
    path = get_file('crawl-300d-2M.vec.zip',
                    'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')
    # crawl-300d-2M.vec?
    vec_file = os.path.join(os.path.dirname(path), 'crawl-300d-2M.vec')
    if not os.path.exists(vec_file):
        zipfile.ZipFile(path).extractall(os.path.dirname(path))
    assert(os.path.exists(vec_file))
    return vec_file

local_USE_dan = None
local_USE_large = None
local_InferSent = None

def sentence_embed(embed_name, sentences, batch_size=None):
    """A helper function for simple external usage, with implicitly created embeder module instances."""
    global local_USE_dan
    global local_USE_large
    global local_InferSent
    if embed_name == 'USE':
        if local_USE_dan is None:
            local_USE_dan = UseEmbedder('dan')
        module = local_USE_dan
    elif embed_name == 'USE-Large':
        if local_USE_large is None:
            local_USE_large = UseEmbedder('transformer')
        module = local_USE_large
    elif embed_name == 'InferSent':
        if local_InferSent is None:
            local_InferSent = InferSentEmbedder()
        module = local_InferSent
    else:
        assert False
    return module.embed(sentences, batch_size)



# collect into one arry
def flatten(v):
    """
    [[],[]] to []
    """
    res = []
    for vi in v:
        res.extend(vi)
    return res

def get_shape(v):
    """
    [[],[]] to [4,2]
    """
    assert(type(v) is list)
    res = []
    for vi in v:
        res.append(len(vi))
    return res

def restore_shape(v, shape):
    """v is flat
    """
    res = []
    idx = 0
    for l in shape:
        res.append(v[idx:idx+l])
        idx = idx+l
    return res


def __test():
    v = [[1,2,3], [4,5], [6,7,8,9]]
    assert(flatten(v) == [1,2,3,4,5,6,7,8,9])
    assert(restore_shape(flatten(v), get_shape(v)) == v)

def embed_keep_shape(v, embedder_name):
    flattened = flatten(v)
    shape = get_shape(v)
    batch_size = {'USE': config.USE_BATCH_SIZE,
                  'USE-Large': config.USE_LARGE_BATCH_SIZE,
                  'InferSent': config.INFERSENT_BATCH_SIZE}[embedder_name]
    print('embedding', len(flattened), 'sentences ..')
    embedding_flattened = sentence_embed(embedder_name, flattened,
                                         batch_size=batch_size)
    embedding = restore_shape(embedding_flattened, shape)
    return embedding

def __test():
    sentences = ['Everyone really likes the newest benefits ',
                 'The Government Executive articles housed on the website are not able to be searched . ',
                 'I like him for the most part , but would still enjoy seeing someone beat him . ',
                 'My favorite restaurants are always at least a hundred miles away from my house . ',
                 'I know exactly . ',
                 'We have plenty of space in the landfill . ']

    out = sentence_embed('USE', sentences, batch_size=1024)
    out = sentence_embed('USE-Large', sentences, batch_size=1024)
    out = sentence_embed('InferSent', sentences, batch_size=1024)

class UseEmbedder():
    """This module is outputing:

    INFO:tensorflow:Saver not created because there are no variables
    in the graph to restore

    Quite annoying, so:

    >>> tf.logging.set_verbosity(tf.logging.WARN)
    
    Or:
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    """
    def __init__(self, encoder='transformer', bsize=128, gpu=True):
        assert(encoder in ['transformer', 'dan'])
        self.bsize = bsize
        # FIXME multi GPU
        self.device = '/gpu:0' if gpu else '/cpu:0'
        if encoder == 'transformer':
            url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        print('DEBUG: creating tf hub module from url:', url)
        self.module = hub.load(url)
        print('DEBUG: hub module loaded')
    def embed_impl(self, batch):
        # with tf.device('/cpu:0'):
        # ISSUE: https://github.com/tensorflow/hub/issues/70
        return self.module(batch)
        
    def embed(self, sentences, bsize=None):
        if bsize is None: bsize = self.bsize
        res = []
        print('embedding %s sentences, batch size: %s'
              % (len(sentences), bsize))
        rg = range(0, len(sentences), bsize)
        msg = ''
        start = time.time()
        for idx,stidx in enumerate(rg):
            # print('\b' * len(msg), end='')
            # print('\r')
            if idx % 30 == 0:
                total_time = time.time() - start
                if idx == 0:
                    eta = -1
                else:
                    eta = (total_time / idx) * (len(rg) - idx)
                speed = bsize * idx / total_time
                msg = ('batch size: %s, batch num %s / %s, '
                    'speed: %.0f sent/s, Total Time: %.0fs, ETA: %.0fs'
                    % (bsize,
                        idx, len(rg),
                        speed,
                        total_time,
                        eta))
                print(msg)
            batch = sentences[stidx:stidx + bsize]
            tmp = self.embed_impl(batch)
            res.append(tmp)
        return np.vstack(res)



class InferSentEmbedder():
    def __init__(self, bsize = 256):
        # FIXME this bsize is used only in this class by my code. This
        # is not passed to InferSent
        self.bsize = bsize
        # HACK: https://github.com/lihebi/InferSent
        from infersent.models import InferSent
        # Load our pre-trained model (in encoder/):
        # this bsize seems not used at all
        params_model = {'bsize': 128,
                        # 'bsize': 64,
                        'word_emb_dim': 300,
                        'enc_lstm_dim': 2048, 'pool_type': 'max',
                        'dpout_model': 0.0,
                        # must use the v2 model in INFERSENT_MODEL_PATH
                        'version': 2}
        # The first time model.cuda() throw RuntimeError: cuDNN error:
        # CUDNN_STATUS_EXECUTION_FAILED
        self.infersent = InferSent(params_model)
        self.infersent.cuda()
        # try:
        #     self.infersent.cuda()
        # except RuntimeError:
        #     print('Warning: RuntimeError occured when creating model.',
        #           'This is known bug. Trying one more time.')
        #     self.infersent.cuda()
        # print("Is cuda? %s" % self.infersent.is_cuda())
        self.infersent.load_state_dict(torch.load(get_infersent_modelpath()))
        self.infersent.set_w2v_path(get_infersent_w2vpath())
        # FIXME load K will probably lose some words. E.g. Vocab size : 100000
        # embedding 93882 sentences ..
        # Nb words kept : 3185922/4231554 (75.3%)
        self.loadk()
        pass
    def build_vocab(self, sentences):
        raise Exception('Deprecated')
        # Build the vocabulary of word vectors (i.e keep only those needed):
        # FIXME this should be all sentences
        self.infersent.build_vocab(sentences, tokenize=True)

    def loadk(self):
        # Just load k ..
        self.infersent.build_vocab_k_words(K=100000)
        
    def embed_impl(self, sentences):
        # Encode your sentences (list of n sentences):
        # embeddings = self.infersent.encode(sentences, tokenize=True)
        embeddings = self.infersent.encode(sentences, bsize=128,
                                           tokenize=False, verbose=False)
        # This outputs a numpy array with n vectors of dimension
        # 4096. Speed is around 1000 sentences per second with batch
        # size 128 on a single GPU.
        return embeddings
    def embed(self, sentences, bsize=None):
        """This bsize is the data bsize, not the model bsize."""
        # FIXME if bsize is different than self.bsize
        if bsize == None: bsize = self.bsize
        res = []
        print('embedding %s sentences, batch size: %s'
              % (len(sentences), bsize))
        rg = range(0, len(sentences), bsize)
        msg = ''
        start = time.time()
        for idx,stidx in enumerate(rg):
            # I have to set the batch size really small to avoid
            # memory or assertion issue. Thus there will be many batch
            # iterations. The update of python shell buffer in Emacs
            # is very slow, thus only update this every severl
            # iterations.
            if idx % 300 == 0:
                # print('\b' * len(msg), end='')
                # print('\r')
                if idx == 0:
                    eta = -1
                else:
                    eta = ((time.time() - start) / idx) * (len(rg) - idx)
                speed = bsize * idx / (time.time() - start)
                msg = ('batch size: %s, batch num %s / %s, '
                       'speed: %.0f sent/s, ETA: %.0fs'
                       % (bsize,
                          idx, len(rg),
                          # time.time() - start,
                          speed,
                          eta))
                print(msg)
            batch = sentences[stidx:stidx + bsize]
            tmp = self.embed_impl(batch)
            res.append(tmp)
        return np.vstack(res)
        
