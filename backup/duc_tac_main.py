import json
import keras
import itertools
import functools
import os
import pickle
import numpy as np
import tensorflow as tf
from scipy import stats
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from antirouge import config

# config.ARTICLE_MAX_SENT = 39
# config.SUMMARY_MAX_SENT = 5

def read_group(group):
    # Use USE data for now
    # the data should be [doc, abs, score]
    doc_ids = [d[0] for d in group]
    abs_ids = [d[1] for d in group]
    scores = [d[2] for d in group]
    def get_doc_content(docID):
        fname = os.path.join(config.DUC_2002_DIR, 'text', docID + '.txt')
        with open(fname) as fp:
            return fp.read()
    def get_doc_embedding(docID):
        """FIXME fixed USE embedder."""
        fname = os.path.join(config.DUC_2002_DIR,
                             'text_embedded', 'USE-Large', docID + '.pkl')
        with open(fname, 'rb') as fp:
            return pickle.load(fp)
    doc_texts = [get_doc_content(ID) for ID in doc_ids]
    doc_USEs = [get_doc_embedding(ID) for ID in doc_ids]
    abs_texts = [get_doc_content(ID) for ID in abs_ids]
    abs_USEs = [get_doc_embedding(ID) for ID in abs_ids]

    np.array(abs_USEs).shape
    # padding
    dtype = doc_USEs[0].dtype
    
    articles = pad_sequences(doc_USEs, value=0, padding='post',
                             maxlen=config.ARTICLE_MAX_SENT,
                             dtype=dtype)
    summaries = pad_sequences(abs_USEs, value=0, padding='post',
                              maxlen=config.SUMMARY_MAX_SENT,
                              dtype=dtype)
    return articles, summaries, scores

def __duc_meta_inspect():
    with open(os.path.join(config.DUC_2002_DIR, 'meta.json')) as fp:
        duc_meta = json.load(fp)
    doc_ids = [d[0] for d in duc_meta]
    abs_ids = [d[1] for d in duc_meta]
    # find the median for doc and abs
    def get_doc_embedding(docID):
        """FIXME fixed USE embedder."""
        fname = os.path.join(config.DUC_2002_DIR,
                             'text_embedded', 'USE', docID + '.pkl')
        with open(fname, 'rb') as fp:
            return pickle.load(fp)
    doc_USEs = [get_doc_embedding(ID) for ID in doc_ids]
    doc_num_sents = [x.shape[0] for x in doc_USEs]
    np.median(doc_num_sents)    # 25
    np.percentile(doc_num_sents, 80)  # 39
    abs_USEs = [get_doc_embedding(ID) for ID in abs_ids]
    abs_num_sents = [x.shape[0] for x in abs_USEs]
    np.median(abs_num_sents)    # 4
    np.percentile(abs_num_sents, 80)  # 5

    
    # check the data
    manual_abs = [d for d in duc_meta if is_manual_abs(d[1])]  # 283
    baseline_abs = [d for d in duc_meta if is_baseline_abs(d[1])]  # 291
    system_abs = [d for d in duc_meta if is_system_abs(d[1])]      # 3816
    assert(len(manual_abs)+ len(baseline_abs)+ len(system_abs) == len(duc_meta))
    print('Num of Abs: Manual: %s, Baseline %s, System %s' %
          (len(manual_abs), len(baseline_abs), len(system_abs)))

    # [d[2] for d in manual_abs]
    np.mean([d[2] for d in manual_abs])  # 0.50
    np.mean([d[2] for d in system_abs])  # 0.31
    np.mean([d[2] for d in baseline_abs])  # 0.37

def is_manual_abs(absID):
    ID = absID.split('--')[1]
    return ID.isalpha() and ID >= 'A' and ID <= 'J'

def is_baseline_abs(absID):
    ID = absID.split('--')[1]
    return ID.isdigit() and int(ID) >=1 and int(ID) <= 3
def is_system_abs(absID):
    ID = absID.split('--')[1]
    return ID.isdigit() and int(ID) >= 15 and int(ID) <= 31


def read_data():
    """
    @return (training, validation, testing), each with (doc, abs, score)
    """
    # load all DUC/TAC data into memory
    # directly train the regression model
    with open(os.path.join(config.DUC_2002_DIR, 'meta.json')) as fp:
        duc_meta = json.load(fp)

    # remove model and manual abs, only evaluate system absID
    # duc_meta = [d for d in duc_meta if is_system_abs(d[1])]
    # duc_meta = [d for d in duc_meta if is_manual_abs(d[1])]
    
    # group data by docID
    # split into training, validation, testing data
    groups = []
    for k, g in itertools.groupby(sorted(duc_meta, key=lambda d: d[0]),
                                  lambda d: d[0]):
        groups.append(list(g))
    len(groups)

    num_groups = len(groups)
    split_1 = int(num_groups * 0.8)
    split_2 = int(num_groups * 0.9)
    training_groups = groups[:split_1]
    validation_groups = groups[split_1:split_2]
    testing_groups = groups[split_2:]
    print('training groups: %s' % len(training_groups))
    print('validation groups: %s' % len(validation_groups))
    print('testing groups: %s' % len(testing_groups))

    def concat_groups(groups):
        a = np.array([x for g in groups for x in g[0]])
        b = np.array([x for g in groups for x in g[1]])
        c = np.array([x for g in groups for x in g[2]])
        return a,b,c

    # (3481, 25, 512)
    training = concat_groups([read_group(g) for g in training_groups])
    # (3481, 4, 512)
    validation = concat_groups([read_group(g) for g in validation_groups])
    # (3481,)
    testing = concat_groups([read_group(g) for g in testing_groups])
    training
    training[0].shape
    training[1].shape
    training[2].shape
    validation
    testing
    return training, validation, testing
    
    # I can directly train on these data

def augment_negative_sampling(data, neg_size):
    """Data is ([doc], [abs], [score]). Assume all manual written."""
    # 1. ignore scores, use 1 instead
    # 2. create negative samples
    docs, abses, scores = data
    # assume one abs per doc
    assert(len(set(docs)) == len(docs))
    augmented = []
    for i in range(len(docs)):
        negative_indices = list(range(i)) + list(range(i+1, len(docs)))
        samples_indices = random.sample(negative_indices, neg_size)
        doc = docs[i]
        abs = abses[i]
        augmented.append((doc, abs, 1))
        for j in samples_indices:
            augmented.append((doc, abses[j], 0))
    random.shuffle(augmented)
    res_docs = [x[0] for x in augmented]
    res_abses = [x[0] for x in augmented]
    res_scores = [x[0] for x in augmented]
    return (res_docs, res_abses), res_scores


def get_sent_model(embedding_size):
    """Sentence embedding model. The size is 512 for USE and 4096 for
InferSent."""
    # embedding_size = 512
    article_input = keras.layers.Input(shape=(config.ARTICLE_MAX_SENT,
                                              embedding_size),
                                       dtype='float32')
    summary_input = keras.layers.Input(shape=(config.SUMMARY_MAX_SENT,
                                              embedding_size),
                                       dtype='float32')
    x = keras.layers.concatenate([article_input, summary_input], axis=1)
    
    # x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(3)(x)
    # x = keras.layers.GlobalMaxPooling1D()(x)
    
    
    # hidden_size = round(K.int_shape(x)[1] / 2)
    # if hidden_size > 128:
    #     hidden_size = 128
    # x= keras.layers.LSTM(hidden_size)(x)

    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    
    preds = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=[article_input, summary_input],
                               outputs=preds)
    return model


def pearson_correlation_f(y_true, y_pred):
    #being K.mean a scalar here, it will be automatically subtracted
    #from all elements in y_pred
    def my_func(y1, y2):
        return stats.pearsonr(y1, y2)[0]
    return tf.py_func(my_func, [y_true, y_pred], tf.float32)
def pearson_correlation_f2(y_true, y_pred):
    #being K.mean a scalar here, it will be automatically subtracted
    #from all elements in y_pred
    fsp = y_pred - K.mean(y_pred)
    fst = y_true - K.mean(y_true)

    devP = K.std(y_pred)
    devT = K.std(y_true)
    return K.mean(fsp*fst)/(devP*devT)
def spearman_correlation_f(y_true, y_pred):
    #being K.mean a scalar here, it will be automatically subtracted
    #from all elements in y_pred
    def my_func(y1, y2):
        # y1, y2 = ys
        return stats.spearmanr(y1, y2)[0]
    return tf.py_func(my_func, [y_true, y_pred], tf.float64)

def exp():
    # Data
    training, validation, testing = read_data()
    training[0].shape
    # Model
    model = get_sent_model(512)
    
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # loss = 'binary_crossentropy'
    loss = 'mse'
    # metrics=['accuracy']
    metrics = ['mae', 'mse', 'accuracy', pearson_correlation_f,
               pearson_correlation_f2,
               spearman_correlation_f]
    # metrics = ['mae', 'mse', 'accuracy',
    #            spearman_correlation_f
    #            # tf.contrib.metrics.streaming_pearson_correlation
    #            # tf.contrib.metrics.streaming_pearson_correlation
    # ]
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
    model.fit([training[0], training[1]], training[2], epochs=100,
              shuffle=True, validation_data=([validation[0],
                                              validation[1]], validation[2]),
              callbacks=[es, mc])
    # saved_model = load_model('best_model.h5')
    model.load_weights('best_model.ckpt')
    loss, mae, mse, _, pearson, pearson2, spearman = model.evaluate([testing[0], testing[1]], testing[2])
    res = model.predict([testing[0], testing[1]])
    testing[2]
    plt.hist(res)
    plt.plot(res)
    plt.hist(testing[2])
    plt.plot(testing[2])
    plt.show()

    print('Testing loss %s, mae %s, mse %s, pearson (%s, %s), spearman %s' %
          (loss, mae, mse, pearson, pearson2, spearman))
    
def __test():
    exp()
