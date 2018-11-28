import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import random
import json

import matplotlib.pyplot as plt

# why keras.preprocessing is not exporting tokenizer_from_json ??
from keras_preprocessing.text import tokenizer_from_json

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras import layers

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

import os, sys

sys.path.append('/home/hebi/github/reading/bert')
sys.path.append('/home/hebi/github/scratch/deeplearning/anti-rouge')

import tokenization
from vocab import Vocab, PAD_TOKEN

sys.path.append('/home/hebi/github/scratch/deeplearning/keras')
from embedding import load_embedding

# vocab_file = '/home/hebi/Downloads/uncased_L-12_H-768_A-12/vocab.txt'
vocab_file = '/home/hebi/github/reading/cnn-dailymail/finished_files/vocab'

cnndm_dir = '/home/hebi/github/reading/cnn-dailymail/'
cnn_dir = os.path.join(cnndm_dir, 'data/cnn/stroies')
dm_dir = os.path.join(cnndm_dir, 'data/dailymail/stories')
cnn_tokenized_dir = os.path.join(cnndm_dir, 'cnn_stories_tokenized')
dm_tokenized_dir = os.path.join(cnndm_dir, 'dm_stories_tokenized')

# this seems to be not useful at all
MAX_NUM_WORDS = 200000

MAX_ARTICLE_LENGTH = 512
MAX_SUMMARY_LENGTH = 128

def tokenize_cnndm():
    # using bert tokenizer
    tokenizer = FullTokenizer(vocab_file=vocab_file)
    tokens = tokenizer.tokenize('hello world! This is awesome.')
    for f in os.listdir(cnn_dir):
        file = os.path.join(cnn_dir, f)
        tokenized_dir = os.path.join(cnn_dir, )
    tokenizer.tokenize()

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def get_art_abs(story_file):
    lines = read_text_file(story_file)
    lines = [line.lower() for line in lines]
    # lines = [fix_missing_period(line) for line in lines]
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    article = ' '.join(article_lines)
    abstract = ' '.join(highlights)
    return article, abstract

def delete_words(summary, ratio):
    words = summary.split(' ')
    length = len(words)
    indices = set(random.sample(range(length),
                                int((1 - ratio) * length)))
    return ' '.join([words[i] for i in range(length)
                     if i not in indices])


def add_words(summary, ratio, vocab):
    words = summary.split(' ')
    length = len(words)
    indices = set([random.randint(0, length)
                   for _ in range(int((1 - ratio) * length))])
    res = []
    for i in range(length):
        if i in indices:
            res.append(vocab.random_word())
        res.append(words[i])
    return ' '.join(res)


def mutate_summary(summary, vocab):
    """I need to generate random mutation to the summary. Save it to a
    file so that I use the same generated data. For each summary, I
    generate several data:
        
    1. generate 10 random float numbers [0,1] as ratios
    2. for each ratio, do:
    2.1 deletion: select ratio percent of words to remove
    2.2 addition: add ratio percent of new words (from vocab.txt) to
    random places

    Issues:
    
    - should I add better, regularized noise, e.g. gaussian noise? How
      to do that?
    - should I check if the sentence is really modified?
    - should we use the text from original article?
    - should we treat sentences? should we maintain the sentence
      separator period?

    """
    ratios = [random.random() for _ in range(10)]
    res = []
    # add the original summary
    res.append([summary, 1.0, 'orig'])
    # the format: ((summary, score, mutation_method))
    for r in ratios:
        s = delete_words(summary, r)
        res.append((s, r, 'del'))
        s = add_words(summary, r, vocab)
        res.append((s, r, 'add'))
    return res

def preprocess_data():
    """
    1. load stories
    2. tokenize
    3. separate article and summary
    4. chunk and save

    This runs pretty slow
    """
    print('Doing nothing.')
    return 0
    vocab = Vocab(vocab_file, 200000)

    # 92,579 stories
    stories = os.listdir(cnn_tokenized_dir)
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    if not os.path.exists(hebi_dir):
        os.makedirs(hebi_dir)
    # hebi/xxxxxx/article.txt
    # hebi/xxxxxx/summary.json
    ct = 0
    for s in stories:
        ct += 1
        # if ct > 10:
        #     return
        # print('--', ct)
        if ct % 100 == 0:
            print ('--', ct*100)
        f = os.path.join(cnn_tokenized_dir, s)
        article, summary = get_art_abs(f)
        pairs = mutate_summary(summary, vocab)
        # write down to file
        d = os.path.join(hebi_dir, s)
        if not os.path.exists(d):
            os.makedirs(d)
        article_f = os.path.join(d, 'article.txt')
        summary_f = os.path.join(d, 'summary.json')
        with open(article_f, 'w') as fout:
            fout.write(article)
        with open(summary_f, 'w') as fout:
            json.dump(pairs, fout, indent=4)

def encode(sentence, vocab):
    """Using vocab encoding
    """
    words = sentence.split()
    return [vocab.word2id(w) for w in words]

def encode(sentence, vocab):
    """Using sentence encoding
    """
    words = sentence.split()
    return [vocab.word2id(w) for w in words]

def encode(sentence, vocab):
    """Using word embedding
    """
    words = sentence.split()
    return [vocab.word2id(w) for w in words]

def decode(ids, vocab):
    return ' '.join([vocab.id2word(i) for i in ids])

def load_text_data():
    """Return (articles, summaries, scores)
    """
    articles = []
    summaries = []
    scores = []
    # 90,000 > 2,000
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    hebi_sample_dir = os.path.join(cnndm_dir, 'hebi-sample')
    data_dir = hebi_sample_dir
    # data_dir = hebi_dir
    ct = 0
    s = os.listdir(data_dir)[0]
    for s in os.listdir(data_dir):
        ct+=1
        if ct % 100 == 0:
            print ('--', ct)
        article_f = os.path.join(data_dir, s, 'article.txt')
        summary_f = os.path.join(data_dir, s, 'summary.json')
        summary_f
        article_content = ' '.join(read_text_file(article_f))
        with open(summary_f, 'r') as f:
            j = json.load(f)
            for summary,score,_ in j:
                articles.append(article_content)
                summaries.append(summary)
                scores.append(score)
    return (articles, summaries, scores)

def test_keras_preprocessing():
    tokenizer.texts_to_sequences(['hello world you are awesome', 'you is good person'])
    tokenizer.sequences_to_texts([[10468, 88, 35, 22, 6270], [35, 11, 199, 363]])
    one_hot('hello world you are you awesome', 200)

def prepare_tokenizer(texts):
    """Tokenizer needs to fit on the given text. Then, we can use it to
    obtain:

    1. tokenizer.texts_to_sequences (texts)
    2. tokenizer.word_index

    """
    # finally, vectorize the text samples into a 2D integer tensor
    # num_words seems to be not useful at all. I set it to 20,000, but
    # the tokenizer.index_word is still 178,781. I set it to 200,000
    # to be consistent
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def prepare_data_v2(articles, summaries, scores, tokenizer):
    print('article texts to sequences ..')
    # TODO this is pretty slow. I parsed each article 21 times. I can
    # surely reduce this
    article_sequences = tokenizer.texts_to_sequences(articles)
    print('padding ..')
    article_sequences_padded = pad_sequences(article_sequences,
                                             value=0, padding='post',
                                             maxlen=MAX_ARTICLE_LENGTH)
    print('summary texts to sequences ..')
    summary_sequences = tokenizer.texts_to_sequences(summaries)
    print('padding ..')
    summary_sequences_padded = pad_sequences(summary_sequences,
                                             value=0, padding='post',
                                             maxlen=MAX_SUMMARY_LENGTH)
    print('concatenating ..')
    data = np.concatenate((article_sequences_padded,
                           summary_sequences_padded), axis=1)


    # shuffle the order
    print('shuffling ..')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    scores_data = np.array(scores)
    scores_data = scores_data[indices]

    # split the data into a training set and a validation set
    print('splitting ..')
    num_validation_samples = int(0.1 * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = scores_data[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = scores_data[-num_validation_samples:]
    return (x_train, y_train), (x_val, y_val)

import re
def sentence_split(s):
    # FIXME keep the seperator
    res = re.split(r'\.|!|\?', s)
    res = [r.strip() for r in res if r]
    return res

def test():
    test_str = 'hello hello hello . world world ! eh eh eh ? yes yes ... ok ok'
    sentence_split(test_str)


def embedder_test(embedder, articles):
    """Testing the performance of embedder.
    """
    # 7,742,028
    all_sents = []
    for article in articles:
        sents = sentence_split(article)
        all_sents.extend(sents)
        
    t = time.time()
    # 9.186472415924072
    s1000 = all_sents[:1000]
    embedder.embed(s1000)
    print(time.time() - t)
    
    t = time.time()
    # 11.613808870315552
    s5000 = all_sents[:5000]
    embedder.embed(s5000)
    print(time.time() - t)

    t = time.time()
    # 15.440065145492554
    s10000 = all_sents[:10000]
    embedder.embed(s10000)
    print(time.time() - t)

    t = time.time()
    # 49.14992094039917
    s50000 = all_sents[:50000]
    embedder.embed(s50000)
    print(time.time() - t)

def sent_embed_articles(articles, max_sent, embedder):
    """

    Input: list of articles or summaries.

    1. break an article into sentences
    2. sentence encoding sentences into 512-dim vectors
    3. max sentence
    """
    a = articles[0]
    sents = sentence_split(a)
    sents_padded = pad_sequences([sents], value='', padding='post',
                                 maxlen=10, dtype=object)
    sents_padded[0].shape
    list(sents_padded[0])
    embedder.embed(sents_padded[0])
    e = embedder.embed(sents)
    e
    e.shape
    e[0].shape
    [e, e]
    pad_sequences([e, e], value=np.zeros(512), padding='post',
                  dtype=float32, maxlen=18)
    

    all_sents = []
    ct = 0
    # using first 1000 for testing
    for article in articles[:1000]:
        ct += 1
        if ct % 1000 == 0:
            print('--', ct)
        sents = sentence_split(article)
        sents_padded = pad_sequences([sents], value='',
                                     padding='post', dtype=object,
                                     maxlen=10)[0]
        # the shape is (#sent, 512)
        # sents_data = embedder.embed(sents)
        # res.append(sents_data)
        res.extend(sents_padded)
    # TODO NOW
    embedded = embedder.embed(res)
    # shape: (#article, 10/3, 512)
    res_padded = pad_sequences(res, value=np.zeros(512),
                               padding='post', maxlen=max_sent)
    return res_padded

def split_sent_and_pad(articles, maxlen):
    res = []
    for article in articles:
        sents = sentence_split(article)
        sents_data = pad_sequences([sents], value='', padding='post',
                                   dtype=object, maxlen=maxlen)[0]
        # the shape is (#sent, 512)
        res.append(sents_data)
    return np.array(res)
    

def prepare_data_v4(articles, summaries, scores):
    """
    Return a padded sequence of sentences.

    (#article, 10+3, string)
    """
    # (#article, 10)
    print('processing articles ..')
    article_data = split_sent_and_pad(articles, 10)
    article_data.shape
    # (#article, 3)
    print('processing summaries ..')
    summary_data = split_sent_and_pad(summaries, 3)
    summary_data.shape
    print('connecting ..')
    data = np.concatenate((article_data, summary_data), axis=1)
    # (#article, 13)
    data.shape
    
    # shuffle the order
    print('shuffling ..')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    scores_data = np.array(scores)
    scores_data = scores_data[indices]

    # split the data into a training set and a validation set
    print('splitting ..')
    num_validation_samples = int(0.1 * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = scores_data[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = scores_data[-num_validation_samples:]
    return (x_train, y_train), (x_val, y_val)

def prepare_data_v3(articles, summaries, scores, tokenizer):
    """
    """
    ARTICLE_MAX_SENT = 10
    SUMMARY_MAX_SENT = 3
    # (#num, 10, 512)
    print('creating sentence embedder instance ..')
    embedder = SentenceEmbedder()
    print('sentence embedding articles ..')
    article_data = sent_embed_articles(articles, ARTICLE_MAX_SENT, embedder)
    # (#num, 3, 512)
    print('sentence embedding summaries ..')
    summary_data = sent_embed_articles(summaries, SUMMARY_MAX_SENT, embedder)
    # concatenate
    print('concatenating ..')
    data = np.concatenate((article_data, summary_data), axis=1)
    
    # shuffle the order
    print('shuffling ..')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    scores_data = np.array(scores)
    scores_data = scores_data[indices]

    # split the data into a training set and a validation set
    print('splitting ..')
    num_validation_samples = int(0.1 * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = scores_data[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = scores_data[-num_validation_samples:]
    return (x_train, y_train), (x_val, y_val)
    

def prepare_data():
    """
    1. define a batch
    2. load a batch
    3. return as (article, summary) pairs?

    1. load all stories and summaries
    2. convert stories and summaries into vectors, according to vocab
    3. trunk or pad with MAX_LEN
    3. return (article, summary, score)
    """
    article_data = []
    summary_data = []
    score_data = []
    # 90,000 > 2,000
    vocab = Vocab(vocab_file, 200000)
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    hebi_sample_dir = os.path.join(cnndm_dir, 'hebi-sample')
    data_dir = hebi_sample_dir
    # data_dir = hebi_dir
    ct = 0
    for s in os.listdir(data_dir):
        ct+=1
        if ct % 100 == 0:
            print ('--', ct)
        article_f = os.path.join(data_dir, s, 'article.txt')
        summary_f = os.path.join(data_dir, s, 'summary.json')
        article_content = ' '.join(read_text_file(article_f))
        article_encoding = encode(article_content, vocab)
        with open(summary_f, 'r') as f:
            summaries = json.load(f)
            for summary,score,_ in summaries:
                article_data.append(article_encoding)
                summary_data.append(encode(summary, vocab))
                score_data.append(score)
    print('converting to numpy array ..')
    score_data = np.array(score_data)
    summary_data = np.array(summary_data)
    article_data = np.array(article_data)

    # plt.plot([len(v) for v in summary_data])
    # plt.hist([len(v) for v in article_data])

    print('padding ..')
    article_data_padded = pad_sequences(article_data,
                                        value=vocab.word2id(PAD_TOKEN),
                                        padding='post',
                                        maxlen=512)
    summary_data_padded = pad_sequences(summary_data,
                                        value=vocab.word2id(PAD_TOKEN),
                                        padding='post',
                                        maxlen=100)
    # x = np.array([np.concatenate((a,s)) for a,s in zip(article_data_padded, summary_data_padded)])
    print('concatenating ..')
    x = np.concatenate((article_data_padded, summary_data_padded),
                       axis=1)
    # np.concatenate((([1,2]), np.array([3])))
    # np.concatenate(([1,2], [3], [4,5,6]))
    y = score_data
    x.shape                     # (21000,768)
    y.shape                     # (21000,)
    # split x and y in training and testing
    split_at = len(x) // 10
    x_train = x[split_at:]
    x_val = x[:split_at]
    y_train = y[split_at:]
    y_val = y[:split_at]
    return (x_train, y_train), (x_val, y_val)

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

def use_main():
    articles, summaries, scores = load_text_data()
    (x_train, y_train), (x_val, y_val) = prepare_data_v4(articles,
                                                         summaries, scores)

    use_embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    model = build_use_model(use_embed)
    use_embed(['hello'])
    # training op
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer=tf.train.AdamOptimizer(0.01)
    model.compile(optimizer=optimizer,
                  # loss='binary_crossentropy',
                  loss='mse',
                  # metrics=['accuracy']
                  metrics=['mae']
    )
    model.fit(x_train, y_train,
              epochs=40, batch_size=128,
              validation_data=(x_val, y_val), verbose=1)
    model.summary()
    

def save_tokenizer(tokenizer):
    # save tokenizer
    j_str = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(j_str)
def load_tokenizer():
    # load
    with open('tokenizer.json') as f:
        j_str = f.read()
        tokenizer = tokenizer_from_json(j_str)
        return tokenizer

import pickle
def save_data(x_train, y_train, x_val, y_val):
    data = [x_train, y_train, x_val, y_val]
    with open('data.json', 'w') as f:
        json.dump(f, data)
def load_data():
    with open('data.json') as f:
        x_train, y_train, x_val, y_val = json.load(f)
        return x_train, y_train, x_val, y_val

    

def main():
    """Steps:
    1. preprocess data:
      - tokenization (sentence tokenizer)
      - separate article and reference summary
      - chunk into train and test

    2. data generation: for each reference summary, do the following
    mutation operations: deletion, insertion, mutation. According to
    how much are changed, assign a score.
    
    3. sentence embedding: embed article and summary into sentence
    vectors. This is the first layer, the embedding layer. Then, do a
    padding to get the vector to the same and fixed dimension
    (e.g. summary 20, article 100). FIXME what to do for very long
    article? Then, fully connected layer directly to the final result.

    """

    # data v1
    (x_train, y_train), (x_val, y_val) = prepare_data()
    # data v2
    articles, summaries, scores = load_text_data()
    # this is pretty time consuming, so save it
    tokenizer = prepare_tokenizer(articles + summaries)
    # alternatively, save and load. Note that you must ensure to fit
    # on the same text.
    save_tokenizer(tokenizer)
    tokenizer = load_tokenizer()
    # this is also slow
    (x_train, y_train), (x_val, y_val) = prepare_data_v2(articles,
                                                         summaries, scores,
                                                         tokenizer)
    # save and load the data
    # save_data(x_train, y_train, x_val, y_val)
    # (x_train, y_train), (x_val, y_val) = load_data()
    
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    # model v1
    model = build_model()
    # model v2
    embedding_layer = load_embedding(tokenizer)
    model = build_glove_model(embedding_layer)
    
    # training op
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer=tf.train.AdamOptimizer(0.01)
    model.compile(optimizer=optimizer,
                  # loss='binary_crossentropy',
                  loss='mse',
                  # metrics=['accuracy']
                  metrics=['mae']
    )
    model.fit(x_train, y_train,
              epochs=40, batch_size=128,
              validation_data=(x_val, y_val), verbose=1)
    model.summary()
    # results = model.evaluate(x_test, y_test)

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
    def __init__(self):
        self.module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.embed_session = tf.Session()
        self.embed_session.run(tf.global_variables_initializer())
        self.embed_session.run(tf.tables_initializer())
    def embed(self, sentence):
        with tf.device('/cpu:0'):
            embedded = self.module(sentence)
        res = self.embed_session.run(embedded)
        return res

def myembed(sentence):
    embedder = SentenceEmbedder()
    embedder.embed(sentence)
    """Embed a string into 512 dim vector
    """
    sentence = ["The quick brown fox jumps over the lazy dog."]
    sentence = ["The quick brown fox is a jumping dog."]
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    # embeddings = embed(["The quick brown fox jumps over the lazy dog."])
    embed_session = tf.Session()
    embed_session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    with tf.device('/cpu:0'):
        embedded = embed(sentence)
    res = embed_session.run(embedded)
    return res

def test():
    myembed(["The quick brown fox jumps over the lazy dog."])
    with tf.device('/cpu:0'):
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        embeddings = embed(["The quick brown fox jumps over the lazy dog."])
        session = tf.Session()
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedded = session.run(embeddings)
        print (embedded)
    pass

def main():
    with tf.device('/cpu:0'):
        sentence_embedding()
