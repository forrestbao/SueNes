# how does tensorflow dataset generate 3 splits. random? 
# truncate of string 

import tensorflow as tf 
import tensorflow_hub as hub
import time
import numpy as np

class UseEmbedder():
    """This module is outputing:
    INFO:tensorflow:Saver not created because there are no variables
    in the graph to restore
    Quite annoying, so:
    >>> tf.logging.set_verbosity(tf.logging.WARN)
    
    Or:
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    Modified from Hebi's code

    """
    def __init__(self, encoder='transformer', bsize=128, gpu=True):
        assert(encoder in ['transformer', 'dan'])
        self.bsize = bsize
        # FIXME multi GPU
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
    def embed_impl(self, batch):
        # with tf.device('/cpu:0'):
        # ISSUE: https://github.com/tensorflow/hub/issues/70
#        with tf.device(self.device):
            embedded = self.module(batch)
            tmp = self.embed_session.run(embedded)
            return tmp
        
    def embed(self, sentences):
        res = []
        print('embedding %s sentences, batch size: %s'
              % (len(sentences), self.bsize))
        rg = range(0, len(sentences), self.bsize)
        msg = ''
        start = time.time()
        for idx,stidx in enumerate(rg):
            print('\b' * len(msg), end='')
            # print('\r')
            total_time = time.time() - start
            if idx == 0:
                eta = -1
            else:
                eta = (total_time / idx) * (len(rg) - idx)
            speed = self.bsize * idx / total_time
            msg = ('batch size: %s, batch num %s / %s, '
                   'speed: %.0f sent/s, Total Time: %.0fs, ETA: %.0fs'
                   % (self.bsize,
                      idx, len(rg),
                      speed,
                      total_time,
                      eta))
            print(msg, end='', flush=True)
            batch = sentences[stidx:stidx + self.bsize]
            tmp = self.embed_impl(batch)
            res.append(tmp)
        print('')
        return np.vstack(res)
    def close(self):
        self.embed_session.close()


def sent_embed(datapairs, sentence_encoder):
    """Embed one sentence into a vector, and save the data into the format for NN training

    datapairs: 3-tuples (str, str, int/float) for 
                _doc, _sum, and label, respectively 

    return: list of 3-tuples, (list of 1-D array, list of 1-D array, int/float) for
            vectors of _doc, vectors of _sum, and label

    """
   
    pass




def test_USE():
    """Embed ai string into 512 dim vector

    Modified from Hebi's code 

    """
    sentences = ['Everyone really likes the newest benefits ',
                 'The Government Executive articles housed on the website are not able to be searched . ',
                 'I like him for the most part , but would still enjoy seeing someone beat him . ',
                 'My favorite restaurants are always at least a hundred miles away from my house . ',
                 'I know exactly . ',
                 'We have plenty of space in the landfill . '] * 100000
    embedder = UseEmbedder(encoder='transformer', bsize=int(1024*(2**5)*1.5), gpu=True)
#    embedder = UseEmbedder(encoder='dan', bsize=1024, gpu=False)
    embeddings = embedder.embed(sentences)
    embeddings.shape
    embeddings

if __name__ == "__main__":
	test_USE()
