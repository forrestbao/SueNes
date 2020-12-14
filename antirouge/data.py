import random
import math
from keras.preprocessing.sequence import pad_sequences
import keras
import os
import pickle
import shutil


import matplotlib.pyplot as plt

import numpy as np
import glob
import tensorflow as tf


from antirouge import config

from antirouge.utils import load_tokenizer, load_tokenizer_CNN
from antirouge.pre import glob_sorted


def ensure_cnn():
    pass

def ensure_dm():
    pass


import tensorflow as tf
import tensorflow_datasets as tfds


class SequenceWrapper(keras.utils.Sequence):
    def __init__(self, files, file_batch_size, batch_size, neg_size):
        self.files = files
        self.file_batch_size = file_batch_size
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.on_epoch_end()
        print('num batch:', len(self.batches))
    def __getitem__(self, index):
        # index is ignored
        x,y,label = next(self.iterator)
        return [x,y], label
    def on_epoch_end(self):
        # random.seed(0)
        random.shuffle(self.files)
        self.batches = Batch(self.files, self.file_batch_size,
                             self.batch_size, self.neg_size)
        self.iterator = self.batches.__iter__()
    def __len__(self):
        return len(self.batches)



class Batch():
    def __init__(self, files, file_batch_size, batch_size,
                 neg_size):
        """
        - file_batch_size deterimines how many files to open at one time (and
          the stories in those files will be shuffled).
        - batch_size determines the returned data sample size.

        We MUST have the file_batch_size * sample_per_file > batch_size, ideally
        it should divide nicely."""
        # print('creating batch ..')
        self.files = files
        self.file_batch_size = file_batch_size
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.tokenizer = None
        # read the first file to determine the length of each file
        # print('reading file length ..')
        assert len(files) >= 1
        with open(files[0], 'rb') as f1:
            self.story_per_file = len(pickle.load(f1))
        # with open(files[0], 'rb') as f1, open(files[1], 'rb') as f2:
        #     self.story_per_file = max(len(pickle.load(f1)), len(pickle.load(f2)))
    def __iter__(self):
        # FIXME if less than file_batch_size, read all rest files
        file_batch_num = int(math.ceil(len(self.files) / self.file_batch_size))
        # HACK will loop one more time if reaches end, so that it won't throw generator error
        for idx in list(range(file_batch_num)) * 2:
            start = self.file_batch_size * idx
            end = self.file_batch_size * (idx+1)
            # Read the files
            files = self.files[start:end]
            stories = []
            # print('reading data %s files ..' % self.file_batch_size)
            for fname in files:
                with open(fname, 'rb') as f:
                    tmp = pickle.load(f)
                    stories.extend(tmp)
            # now shuffle the data
            random.shuffle(stories)

            augmented_stories = []
            articles = [s[1] for s in stories]
            summaries = [s[2] for s in stories]
            # Padding
            if isinstance(articles[0], str):
                if self.tokenizer is None:
                    self.tokenizer = load_tokenizer_CNN()
                articles = self.tokenizer.texts_to_sequences(articles)
                summaries = self.tokenizer.texts_to_sequences(summaries)
                maxlen_article = config.ARTICLE_MAX_WORD
                maxlen_summary = config.SUMMARY_MAX_WORD
            else:
                maxlen_article = config.ARTICLE_MAX_SENT
                maxlen_summary = config.SUMMARY_MAX_SENT
            dtype = np.array(articles[0]).dtype
            articles = pad_sequences(articles, value=0,
                                     padding='post',
                                     maxlen=maxlen_article,
                                     dtype=dtype)
            summaries = pad_sequences(summaries, value=0,
                                      padding='post',
                                      maxlen=maxlen_summary,
                                      dtype=dtype)
            # now generate negative samples
            #
            # FIXME this only supports negative sampling. How about mutation?
            # That would require rerun the sentence embedder.
            #
            # print('generating negative samples ..')
            for i in range(len(stories)):
                negative_indices = list(range(i)) + list(range(i+1,len(stories)))
                samples_indices = random.sample(negative_indices, self.neg_size)
                article = articles[i]
                summary = summaries[i]
                augmented_stories.append((article, summary, 1))
                for x in samples_indices:
                    augmented_stories.append((article,
                                              summaries[x], 0))
            # print('shuffling batch ..')
            random.shuffle(augmented_stories)
            # generate batch
            batch_num = int(len(augmented_stories) / self.batch_size)
            for i in range(batch_num):
                start = self.batch_size * i
                end = self.batch_size * (i+1)
                batch_stories = augmented_stories[start:end]
                x = np.array([s[0] for s in batch_stories])
                y = np.array([s[1] for s in batch_stories])
                labels = np.array([s[2] for s in batch_stories])
                yield (x,y,labels)
            
    def __len__(self):
        # FIXME this may not be accurate
        return int((len(self.files)) * self.story_per_file *
                   (self.neg_size + 1) / self.batch_size)

def get_data_generators(folder):
    """For old CNN data folder"""
    files = glob_sorted(folder + '/*')
    random.shuffle(files)
    # files
    # DEBUG simulating previous small data
    # files = files[:int(len(files)/3)]
    num_files = len(files)
    split_1 = int(num_files * 0.8)
    split_2 = int(num_files * 0.9)
    split_1
    training_files = files[:split_1]
    testing_files = files[split_1:split_2]
    validating_files = files[split_2:]
    print('training files: %s' % len(training_files))
    print('validation files: %s' % len(validating_files))
    print('testing files: %s' % len(testing_files))
    # CAUTION 3 * story_per_file should be larger than 100 ..
    training_seq = SequenceWrapper(training_files, 3, 100, config.NEG_SIZE)
    validation_seq = SequenceWrapper(validating_files, 3, 100, config.NEG_SIZE)
    testing_seq = SequenceWrapper(testing_files, 3, 100, config.NEG_SIZE)
    return training_seq, validation_seq, testing_seq


def get_separate_generators(train_folder, validation_folder, test_folder, bsize=100):
    """For new xxx_add/cross/delete/replace folders"""
    training_files = glob_sorted(train_folder + '/*')
    testing_files = glob_sorted(test_folder + '/*')
    validating_files = glob_sorted(validation_folder + '/*')
    print('training files: %s' % len(training_files))
    print('validation files: %s' % len(validating_files))
    print('testing files: %s' % len(testing_files))
    # CAUTION 3 * story_per_file should be larger than 100 ..
    training_seq = SequenceWrapper(training_files, 3, bsize, config.NEG_SIZE)
    validation_seq = SequenceWrapper(validating_files, 3, bsize, config.NEG_SIZE)
    testing_seq = SequenceWrapper(testing_files, 3, bsize, config.NEG_SIZE)
    return training_seq, validation_seq, testing_seq