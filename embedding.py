#!/usr/bin/env python3

import sys

import tensorflow as tf
import tensorflow_hub as hub
import torch
import numpy as np

from config import *

class UseEmbedder():
    """This module is outputing:

    INFO:tensorflow:Saver not created because there are no variables
    in the graph to restore

    Quite annoying, so:

    >>> tf.logging.set_verbosity(logging.WARN)
    
    Or:
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    """
    def __init__(self, encoder='transformer', bsize=128, gpu=True):
        assert(encoder in ['transformer', 'dan'])
        self.bsize = bsize
        self.device = '/gpu:0' if gpu else '/cpu:0'
        if encoder == 'transformer':
            url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        else:
            url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.module = hub.Module(url)
        config = tf.ConfigProto(allow_soft_placement = True)
        self.embed_session = tf.Session(config = config)
        # self.embed_session = tf.Session()
        self.embed_session.run(tf.global_variables_initializer())
        self.embed_session.run(tf.tables_initializer())
    def embed(self, sentences):
        # with tf.device('/cpu:0'):
        # ISSUE: https://github.com/tensorflow/hub/issues/70
        res = []
        for stidx in range(0, len(sentences), self.bsize):
            batch = sentences[stidx:stidx + self.bsize]
            with tf.device(self.device):
                embedded = self.module(batch)
                tmp = self.embed_session.run(embedded)
                res.append(tmp)
        return np.vstack(res)
    def close(self):
        self.embed_session.close()

# FIXME move this to config
sys.path.append('/home/hebi/github/reading/InferSent/')

class InferSentEmbedder():
    def __init__(self):
        # Load our pre-trained model (in encoder/):
        from models import InferSent
        # this bsize seems not used at all
        params_model = {'bsize': 256,
                        # 'bsize': 64,
                        'word_emb_dim': 300,
                        'enc_lstm_dim': 2048, 'pool_type': 'max',
                        'dpout_model': 0.0,
                        # must use the v2 model in INFERSENT_MODEL_PATH
                        'version': 2}
        # The first time model.cuda() throw RuntimeError: cuDNN error:
        # CUDNN_STATUS_EXECUTION_FAILED
        self.infersent = InferSent(params_model)
        try:
            self.infersent.cuda()
        except RuntimeError:
            print('Warning: RuntimeError occured when creating model.',
                  'This is known bug. Trying one more time.')
            self.infersent.cuda()
        print("Is cuda? %s" % self.infersent.is_cuda())
        self.infersent.load_state_dict(torch.load(INFERSENT_MODEL_PATH))
        self.infersent.set_w2v_path(INFERSENT_W2V_PATH)
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
        
    def embed(self, sentences):
        # Encode your sentences (list of n sentences):
        # embeddings = self.infersent.encode(sentences, tokenize=True)
        embeddings = self.infersent.encode(sentences, bsize=256,
                                           tokenize=False, verbose=True)
        # This outputs a numpy array with n vectors of dimension
        # 4096. Speed is around 1000 sentences per second with batch
        # size 128 on a single GPU.
        return embeddings

def test_infersent():
    sentences = ['Everyone really likes the newest benefits ',
                 'The Government Executive articles housed on the website are not able to be searched . ',
                 'I like him for the most part , but would still enjoy seeing someone beat him . ',
                 'My favorite restaurants are always at least a hundred miles away from my house . ',
                 'I know exactly . ',
                 'We have plenty of space in the landfill . '] * 10000


    embedder = InferSentEmbedder()
    embeddings = embedder.embed(sentences)
    embeddings
    # import pickle
    # pickle.load(open('encoder/infersent2.pkl', 'rb'))
    # import nltk
    # nltk.download('punkt')


def test_USE():
    """Embed a string into 512 dim vector
    """
    sentences = ["The quick brown fox jumps over the lazy dog."]
    sentences = ['Everyone really likes the newest benefits ',
                 'The Government Executive articles housed on the website are not able to be searched . ',
                 'I like him for the most part , but would still enjoy seeing someone beat him . ',
                 'My favorite restaurants are always at least a hundred miles away from my house . ',
                 'I know exactly . ',
                 'We have plenty of space in the landfill . '] * 10000
    embedder = UseEmbedder(encoder='transformer', bsize=10240, gpu=True)
    embedder = UseEmbedder(encoder='dan', bsize=51200, gpu=False)
    embeddings = embedder.embed(sentences)
    embeddings.shape
    embeddings
