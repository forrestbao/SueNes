import random
from keras.preprocessing.sequence import pad_sequences
import keras
import os
import pickle
import shutil

import matplotlib.pyplot as plt

import numpy as np
import glob
import tensorflow as tf

# from antirouge.config import *

from antirouge import config

from antirouge.serial_preprocess import glob_sorted, load_tokenizer_CNN
from antirouge.utils import load_tokenizer
from antirouge.embedding import load_glove_layer

class SequenceWrapper(keras.utils.Sequence):
    def __init__(self, files, file_batch_size, batch_size, neg_size):
        self.files = files
        self.file_batch_size = file_batch_size
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.on_epoch_end()
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
        # print('creating batch ..')
        self.files = files
        self.file_batch_size = file_batch_size
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.tokenizer = None
        # read the first file to determine the length of each file
        # print('reading file length ..')
        assert len(files) > 1
        with open(files[0], 'rb') as f1, open(files[1], 'rb') as f2:
            self.story_per_file = max(len(pickle.load(f1)), len(pickle.load(f2)))
    def __iter__(self):
        file_batch_num = int(len(self.files) / self.file_batch_size)
        for idx in range(file_batch_num):
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
            # now generate negative samples
            # print('generating negative samples ..')
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
        return int((len(self.files) - 1) * self.story_per_file *
                   (self.neg_size + 1) / self.batch_size)

def get_data_generators(folder):
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
    training_seq = SequenceWrapper(training_files, 3, 1000, config.NEG_SIZE)
    validation_seq = SequenceWrapper(validating_files, 3, 1000, config.NEG_SIZE)
    testing_seq = SequenceWrapper(testing_files, 3, 1000, config.NEG_SIZE)
    return training_seq, validation_seq, testing_seq

def train_model(model, data_folder):
    training_seq, validation_seq, testing_seq = get_data_generators(data_folder)
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

def exp_word():
    print('Glove exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'story')
    model = get_word_model()
    train_model(model, folder)

def exp_USE():
    print('USE exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'USE')
    model = get_sent_model(512)
    train_model(model, folder)
    training_seq, validation_seq, testing_seq = get_data_generators(folder)
    a = testing_seq.__getitem__(0)
    a
    len(a[0])
    a[0][0].shape
    res = model.predict_generator(testing_seq)
    res.shape
    res.mean()
    np.median(res)
    plt.plot(res)
    plt.plot([1,2,3,4,5])
    plt.hist(res)
    a[1]

    
def exp_USE_Large():
    print('USE Large exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'USE-Large')
    model = get_sent_model(512)
    train_model(model, folder)
def exp_InferSent():
    print('InferSent exp')
    folder = os.path.join(config.CNN_SERIAL_DIR, 'InferSent')
    model = get_sent_model(4096)
    train_model(model, folder)

def __test():
    # glove has 88%
    exp_word()
    # USE and USE-Large both have 98%
    exp_USE()
    exp_USE_Large()
    # infersent seems not working, 50%
    exp_InferSent()

def get_sent_model(embedding_size):
    """Sentence embedding model. The size is 512 for USE and 4096 for
InferSent."""
    article_input = keras.layers.Input(shape=(config.ARTICLE_MAX_SENT, embedding_size),
                                       dtype='float32')
    summary_input = keras.layers.Input(shape=(config.SUMMARY_MAX_SENT, embedding_size),
                                       dtype='float32')
    x = keras.layers.concatenate([article_input, summary_input], axis=1)
    
    # x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(3)(x)
    # x = keras.layers.GlobalMaxPooling1D()(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[article_input, summary_input],
                               outputs=preds)
    return model

def get_word_model():
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

 
def __test():
    # 10 epoches
    for i in range(10):
        print('Epoch %s' % (i+1))
        batches = Batch(training_files, 3, 1000)
        total_batch = len(batch)
        ct=0
        for x,y,label in batch:
            ct+=1
            l, acc = model.train_on_batch([x,y], label)
            if ct % 10 == 0:
                print('--- %s / %s, loss: %s, acc: %s' % (ct, total_batch, l, acc))
    
