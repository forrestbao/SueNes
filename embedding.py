#!/usr/bin/env python3

import sys

import tensorflow as tf
import tensorflow_hub as hub
import torch


def sentence_embedding():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    # Compute a representation for each message, showing various lengths supported.
    word = "Elephant"
    sentence = "I am a sentence for which I would like to get its embedding."
    paragraph = (
        "Universal Sentence Encoder embeddings also support short paragraphs. "
        "There is no hard limit on how long the paragraph is. Roughly, the longer "
        "the more 'diluted' the embedding will be.")
    messages = [word, sentence, paragraph]

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(messages))

        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            print("Message: {}".format(messages[i]))
            print("Embedding size: {}".format(len(message_embedding)))
            message_embedding_snippet = ", ".join(
                (str(x) for x in message_embedding[:3]))
            print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

class SentenceEmbedder():
    """This module is outputing:

    INFO:tensorflow:Saver not created because there are no variables
    in the graph to restore

    Quite annoying, so:

    >>> tf.logging.set_verbosity(logging.WARN)
    """
    def __init__(self):
        # 
        self.module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.embed_session = tf.Session()
        self.embed_session.run(tf.global_variables_initializer())
        self.embed_session.run(tf.tables_initializer())
        # self.module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    def embed(self, sentences):
        with tf.device('/cpu:0'):
            embedded = self.module(sentences)
            res = self.embed_session.run(embedded)
            return res
    def embed_list(self, sentences_list):
        with tf.device('/cpu:0'):
            embedded = [self.module(sentences) for sentences in sentences_list]
            res = self.embed_session.run(embedded)
            return res

class SentenceEmbedderLarge():
    """This module is outputing:

    INFO:tensorflow:Saver not created because there are no variables
    in the graph to restore

    Quite annoying, so:

    >>> tf.logging.set_verbosity(logging.WARN)
    """
    def __init__(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.module = hub.Module(module_url)
        self.embed_session = tf.Session()
        self.embed_session.run(tf.global_variables_initializer())
        self.embed_session.run(tf.tables_initializer())
    def embed(self, sentences):
        with tf.device('/cpu:0'):
            embedded = self.module(sentences)
            res = self.embed_session.run(embedded)
            return res
    def embed_list(self, sentences_list):
        with tf.device('/cpu:0'):
            embedded = [self.module(sentences) for sentences in sentences_list]
            res = self.embed_session.run(embedded)
            return res

sys.path.append('/home/hebi/github/reading/InferSent/')

class InferSentEmbedder():
    def __init__(self):
        # Load our pre-trained model (in encoder/):
        from models import InferSent
        V = 2
        MODEL_PATH = 'encoder/infersent%s.pkl' % V
        params_model = {'bsize': 64, 'word_emb_dim': 300,
                        'enc_lstm_dim': 2048, 'pool_type': 'max',
                        'dpout_model': 0.0, 'version': V}
        self.infersent = InferSent(params_model)
        # use cuda!!
        self.infersent = self.infersent.cuda()
        self.infersent.load_state_dict(torch.load(MODEL_PATH))
        # Set word vector path for the model:
        W2V_PATH = 'dataset/fastText/crawl-300d-2M.vec'
        self.infersent.set_w2v_path(W2V_PATH)
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
        embeddings = self.infersent.encode(sentences, bsize=64,
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
                 'We have plenty of space in the landfill . '] * 1000


    embedder = InferSentEmbedder()
    embeddings = embedder.embed(sentences)
    

def test():
    """Embed a string into 512 dim vector
    """
    sentences = ["The quick brown fox jumps over the lazy dog."]
    sentences = ["The quick brown fox is a jumping dog."]
    embedder = SentenceEmbedder()
    embedder = SentenceEmbedderLarge()
    embeddings = embedder.embed(sentences)
    embeddings.shape

def test():
    myembed(["The quick brown fox jumps over the lazy dog."])
    with tf.device('/cpu:0'):
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        embeddings = embed(["The quick brown fox jumps over the lazy dog."])
        session = tf.Session()
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedded = session.run(embeddings)
        print (embedded)

def main():
    with tf.device('/cpu:0'):
        sentence_embedding()
