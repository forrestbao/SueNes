import sys
sys.path.append("..")

from antirouge import preprocessing as pre
from antirouge import main
import tensorflow as tf

def tokenize():
    pre.preprocess_story_pickle()
    pre.preprocess_word_mutated()
    pre.preprocess_negative_sampling()

def embed():
    pre.preprocess_sentence_embed('InferSent', 'story', 10000, 30000)
    pre.preprocess_sentence_embed('InferSent', 'negative', 10000, 30000)
    pre.preprocess_sentence_embed('InferSent', 'mutated', 10000, 30000)
if __name__ == '__main__':
    # embed()
    main.run_exp2('neg', 'InferSent', 30000, 1, '2-LSTM')
    '''
    article_input = tf.keras.Input(shape=(None, 512), dtype='float32')
    summary_input = tf.keras.Input(shape=(None, 512), dtype='float32')
    x = tf.keras.layers.LSTM(128)(article_input)
    y = tf.keras.layers.LSTM(128)(summary_input)
    z = tf.keras.layers.Dense(1, activation='sigmoid')(tf.keras.layers.concatenate([x, y]))
    tf.keras.Model([article_input, summary_input], z)
    '''