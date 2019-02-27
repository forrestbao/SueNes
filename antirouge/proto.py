import os
import pickle
import shutil

import numpy as np
import glob

from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from antirouge.utils import read_text_file, sentence_split
from antirouge.utils import dict_pickle_read, dict_pickle_read_keys, dict_pickle_write

from antirouge.embedding import sentence_embed, sentence_embed_reset

import random
import tensorflow as tf

from antirouge.preprocessing import get_art_abs, embed_keep_shape

from antirouge.config import *

# FIXME the rest files may not need eager execution, thus this file
# should be stand-alone
# 
# iterating through tensors here need eager evalutaion
# if not tf.executing_eagerly():
#     tf.enable_eager_execution()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def proto_encode_story(key, article, summary):
    feature = {
        'key': _bytes_feature(key.encode('utf-8')),
        'article': _bytes_feature(article.encode('utf-8')),
        'summary': _bytes_feature(summary.encode('utf-8'))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _trunk_story_tfrec(story_tfrec_fname):
    """Shuffle, batch, and trunk stories.
    """
    ds = tf.data.TFRecordDataset(story_tfrec_fname)
    # trunk size: 10000
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(10000)
    ct = 0
    folder = os.path.join(PROTO_DIR, 'story')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for batch in ds:
        ct += 1
        print(ct)
        fname = os.path.join(folder, '%s.tfrec' % ct)
        with tf.python_io.TFRecordWriter(fname) as writer:
            for d in batch.numpy():
                writer.write(d)
def proto_process_story():
    # 92,579 stories
    stories = os.listdir(CNN_TOKENIZED_DIR)
    ct = 0
    # save 10000 stories per file
    #
    # NEW now save in one file. Use another routing to split files if
    # necessary.
    story_tfrec_fname = os.path.join(PROTO_DIR, 'story.tfrec')
    with tf.python_io.TFRecordWriter(story_tfrec_fname) as writer:
        for key in stories:
            ct+=1
            if ct % 10000 == 0:
                print(ct)
            story_file = os.path.join(CNN_TOKENIZED_DIR, key)
            article, summary = get_art_abs(story_file)
            exp = proto_encode_story(key, article, summary)
            writer.write(exp.SerializeToString())
    _trunk_story_tfrec(story_tfrec_fname)

def proto_load_story_ds():
    # TODO load not only stories
    # Load story from tfrec file
    folder = os.path.join(PROTO_DIR, 'story')
    raw_ds = tf.data.TFRecordDataset(glob_sorted(folder + '/*'))
    feature_description = {
        'key': tf.FixedLenFeature([], tf.string, default_value=''),
        'article': tf.FixedLenFeature([], tf.string, default_value=''),
        'summary': tf.FixedLenFeature([], tf.string, default_value='')
    }
    def _my_parse_function(pto):
        # Parse the input tf.Example proto using the dictionary above.
        d = tf.parse_single_example(pto, feature_description)
        # TODO decode the binary string?
        # d['summary'] = d['summary'].decode('utf-8')
        return d
    return raw_ds.map(_my_parse_function)


def glob_sorted(pattern):
    """Sort according to the number in the filename."""
    return sorted(glob.glob(pattern), key=lambda f:
                  int(''.join(filter(str.isdigit, f))))

def proto_load_embed_ds(folder):
    raw_ds = tf.data.TFRecordDataset(glob_sorted(folder + '/*'))
    feature_description = {
        'key': tf.FixedLenFeature([], tf.string, default_value=''),
        'article': tf.FixedLenFeature([], tf.string, default_value=''),
        'summary': tf.FixedLenFeature([], tf.string, default_value='')
    }
    def _my_parse_function(pto):
        # Parse the input tf.Example proto using the dictionary above.
        d = tf.parse_single_example(pto, feature_description)
        # I cannot use pickle here, as they are tensors. Thus, I must
        # remember to load the pickle
        #
        # d['summary'] = pickle.loads(d['summary'])
        # d['article'] = pickle.loads(d['article'])
        return d
    return raw_ds.map(_my_parse_function)

def __test():
    folder = os.path.join(PROTO_DIR, 'USE')
    ds = tf.data.TFRecordDataset([folder])
    ds = ds.map(_my_parse_function)
    sess = tf.Session()
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    elem = sess.run(next_element)
    type(elem)
    len(elem)
    elem
    elem.keys()
    type(elem['summary'])
    ss = pickle.loads(elem['summary'])
    type(ss)
    ss

def __test():
    folder = os.path.join(PROTO_DIR, 'USE')
    ds = proto_load_embed_ds(folder)
    ct=0
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        d = sess.run(next_element)
        article = pickle.loads(d['article'])
        summary = pickle.loads(d['summary'])
        print('--- key\n', d['key'])
        print('--- article\n', article.shape)
        print('--- summary\n', summary)
        
    for d in ds.take(10):
        ct+=1
        x=d
    ct
    x

def __test_shuffle_performance():
    folder = os.path.join(PROTO_DIR, 'InferSent')
    # folder = os.path.join(PROTO_DIR, 'USE')
    ds = proto_load_embed_ds(folder)
    ds = ds.shuffle(30000)
    ds = ds.batch(100)
    ct=0
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            ct+=1
            d = sess.run(next_element)
            if ct % 10 == 0:
                print(ct)
                # print(ct, d['key'])
            # if ct % 10000 == 0:
            #     break
            # article = pickle.loads(d['article'])
            # summary = pickle.loads(d['summary'])
            # print('--- key\n', d['key'])
            # print('--- article\n', article.shape)
            # print('--- summary\n', summary)
    

def proto_story_embed(embedder, limit=None):
    def proto_encode_embed_result(key, article, summary):
        feature = {
            'key': _bytes_feature(key),
            'article': _bytes_feature(pickle.dumps(article)),
            'summary': _bytes_feature(pickle.dumps(summary))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    assert embedder in ['USE', 'USE-Large', 'InferSent']
    folder = os.path.join(PROTO_DIR, embedder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with tf.device('/cpu:0'):
        ds = proto_load_story_ds()
        ds = ds.batch(1000)
        iterator = ds.make_one_shot_iterator()
        next_element = iterator.get_next()
        # This is important, otherwise USE-Large will complain
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config=config) as sess:
            ct = 0
            real_ct = 0
            while True:
                batch = sess.run(next_element)
                ct += 1
                fname = os.path.join(folder, '%s.tfrec' % ct)
                # start from where we left
                if os.path.exists(fname):
                    continue
                if limit is not None and real_ct >= limit:
                    print('reach limit %s, breaking' % limit)
                    break
                print(ct)
                real_ct += 1
                # obtain numpy from batch
                articles = batch['article']
                summaries = batch['summary']
                print('articles: ', len(articles))
                print('split sentences ..')
                # sentence split
                articles = [sentence_split(a) for a in articles]
                summaries = [sentence_split(s) for s in summaries]
                # encode
                print('encoding sentences ..')
                articles = embed_keep_shape(articles, embedder)
                summaries = embed_keep_shape(summaries, embedder)
                print('tfrecord writing to ', fname)
                with tf.python_io.TFRecordWriter(fname) as writer:
                    for k, a, s in zip(batch['key'], articles, summaries):
                        exp = proto_encode_embed_result(k, a, s)
                        writer.write(exp.SerializeToString())
                print('Done')
        
 
def __test():
    proto_process_story()
    # FIXME USE-Large performance is decreasing along the runs
    # 
    # FIXME USE-Large GPU utilization not stable, only used like 1/10
    # of the time

    
    while True:
        proto_story_embed('USE', 3)
        sentence_embed_reset()
        tf.reset_default_graph()

    for _ in range(50):
        proto_story_embed('USE-Large', 1)
        print('resetting graph')
        sentence_embed_reset()
        tf.reset_default_graph()


    # The performance for InferSent is good
    proto_story_embed('InferSent')


    ds = proto_load_story_ds()
    print(ct)


def __test():
    ds = proto_load_story_ds()
    ct=0
    tf.nn.nce_loss
    for d in ds.take(10):
        ct+=1
        x=d
        print()
        print('--- key\n', d['key'])
        print('--- article\n', d['article'])
        print('--- summary\n', d['summary'])
    ct
    x

def __test():
    sess = tf.Session()

    tfrec = tf.data.experimental.TFRecordWriter('test.tfrec')
    np.array(['hello', 'world'])
    ds = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
    ds.output_shapes == tensor_shape.scalar()
    tf.tensor_shape
    
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess.run(next_element)
    
    op = tfrec.write(ds)
    sess.run(op)
  
