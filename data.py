#!/usr/bin/env python3
import pickle
import numpy as np
import os
import json
import re
import math

from keras.preprocessing.sequence import pad_sequences


from preprocessing import read_text_file
from embedding import SentenceEmbedder

cnndm_dir = '/home/hebi/github/reading/cnn-dailymail/'
cnn_dir = os.path.join(cnndm_dir, 'data/cnn/stroies')
dm_dir = os.path.join(cnndm_dir, 'data/dailymail/stories')
cnn_tokenized_dir = os.path.join(cnndm_dir, 'cnn_stories_tokenized')
dm_tokenized_dir = os.path.join(cnndm_dir, 'dm_stories_tokenized')


def uae_pregen():
    """Pre-generate the uae for data folder.
    """
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    data_dir = os.path.join(cnndm_dir, 'hebi-sample-100')
    hebi_uae_dir = os.path.join(cnndm_dir, 'hebi-uae')
    # For each folder (with hash) in hebi_dir, check if hebi_uae_dir
    # contains this folder or not. If not, parse the article and
    # summaries into data.
    stories = os.listdir(data_dir)

    # printing out current progress
    finished_stories = os.listdir(hebi_uae_dir)
    print('total stories:', len(stories))
    print('finished:', len(finished_stories))

    use_embedder = SentenceEmbedder()

    ct = 0

    for s in stories:
        data = {}
        to_encode = []
        scores = []
        print('processing', s, '..')
        story_dir = os.path.join(data_dir, s)
        story_uae_file = os.path.join(hebi_uae_dir, s)
        if not os.path.exists(story_uae_file):
            
            ct += 1
            if ct % 50 == 0:
                # This function returns after processing 50 stories,
                # due to memory reason.
                return
            
            # create the embedding
            article_file = os.path.join(story_dir, 'article.txt')
            summary_file = os.path.join(story_dir, 'summary.json')
            with open(article_file) as f:
                article = f.read()
                to_encode.append(article)
            with open(summary_file) as f:
                j = json.load(f)
                for summary,score,_ in j:
                    to_encode.append(summary)
                    scores.append(score)
            # now, encode to_encode
            # to_encode should have 21 paragraphs
            # split into sentences
            to_encode_array = [sentence_split(a) for a in to_encode]
            def get_shape(vv):
                return [len(v) for v in vv]
            def flatten(vv):
                res = []
                for v in vv:
                    res.extend(v)
                return res
            def restructure(v, shape):
                if not shape:
                    return []
                l = shape[0]
                return [v[:l]] + restructure(v[l:], shape[1:])
            shape = get_shape(to_encode_array)
            flattened = flatten(to_encode_array)
            print('embedding', len(flattened), 'sentences ..')
            embedding_flattened = use_embedder.embed(flattened)
            embedding = restructure(embedding_flattened, shape)
            # this embedding should be (21, 512)
            data = {}
            #(numsent, 512)
            data['article'] = embedding[0]
            data['summary'] = embedding[1:]
            data['score'] = scores
            with open(story_uae_file, 'wb') as f:
                pickle.dump(data, f)

def save_data(data, filename):
    (x_train, y_train), (x_val, y_val) = data
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data

def sentence_split(s):
    # FIXME keep the seperator
    res = re.split(r'\.|!|\?', s)
    res = [r.strip() for r in res if r]
    return res

def test():
    test_str = 'hello hello hello . world world ! eh eh eh ? yes yes ... ok ok'
    sentence_split(test_str)

def load_text_data(size='tiny'):
    """Return (articles, summaries, scores)

    SIZE: tiny, small, medium, large, all
    """
    articles = []
    summaries = []
    scores = []
    # 90,000 > 2,000
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    hebi_sample_dir = os.path.join(cnndm_dir, 'hebi-sample')
    data_dispatcher = {
        # TODO other sizes
        'tiny': os.path.join(cnndm_dir, 'hebi-sample-100'),
        'small': os.path.join(cnndm_dir, 'hebi-sample-1000'),
        'medium': os.path.join(cnndm_dir, 'hebi-sample-10000'),
    }
    data_dir = data_dispatcher[size]
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

def test_keras_preprocessing():
    tokenizer.texts_to_sequences(['hello world you are awesome', 'you is good person'])
    tokenizer.sequences_to_texts([[10468, 88, 35, 22, 6270], [35, 11, 199, 363]])
    one_hot('hello world you are you awesome', 200)
    return

def prepare_data_using_tokenizer(articles, summaries, scores, tokenizer):
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

def embedder_test():
    """Testing the performance of embedder.


    s1000: 5.060769319534302
    s250s*4: 10.278579235076904
    s5000: 7.866734266281128
    s1000s*5: 15.469667434692383
    s10000: 12.214599609375
    s2500s*4: 18.09593439102173
    
    """
    embedder = SentenceEmbedder()
    articles, summaries, scores = load_text_data(size='medium')

    print('all articles:', len(articles))
    # 7,742,028
    all_sents = []
    for article in articles:
        sents = sentence_split(article)
        all_sents.extend(sents)

    print('all sentences:', len(all_sents))
        
    t = time.time()
    # 9.186472415924072
    s1000 = all_sents[:1000]
    embedder.embed(s1000)
    print('s1000:', time.time() - t)

    t = time.time()
    s250s = np.array_split(s1000, 4)
    embedder.embed_list(s250s)
    print('s250s*4:', time.time() - t)
    
    t = time.time()
    # 11.613808870315552
    s5000 = all_sents[:5000]
    embedder.embed(s5000)
    print('s5000:', time.time() - t)

    t = time.time()
    s1000s = np.array_split(s5000, 5)
    embedder.embed_list(s1000s)
    print('s1000s*5:', time.time() - t)

    t = time.time()
    # 15.440065145492554
    s10000 = all_sents[:10000]
    embedder.embed(s10000)
    print('s10000:', time.time() - t)

    t = time.time()
    s2500s = np.array_split(s10000, 4)
    embedder.embed_list(s2500s)
    print('s2500s*4:', time.time() - t)

    t = time.time()
    # 49.14992094039917
    s50000 = all_sents[:50000]
    embedder.embed(s50000)
    print('s50000:', time.time() - t)

    t = time.time()
    s10000s = np.array_split(s50000, 5)
    embedder.embed_list(s10000s)
    print('s10000s*5:', time.time() - t)

    return

def sent_embed_articles(articles, maxlen, use_embedder, batch_size=10000):
    """
    BATCH_SIZE is how many articles to send to the embedder.
    
    Input: list of articles or summaries.

    1. break an article into sentences
    2. sentence encoding sentences into 512-dim vectors
    3. max sentence
    """
    # DEBUG remove this assignments
    # maxlen=10
    # batch_size = 10000
    
    sents = [sentence_split(a) for a in articles]
    sents_padded = pad_sequences(sents, value='', padding='post',
                                 maxlen=maxlen, dtype=object)
    shape = sents_padded.shape
    flattened = np.ndarray.flatten(sents_padded)
    
    splits = np.array_split(flattened, math.ceil(len(flattened) / batch_size))
    print('number of batch:', len(splits))
    ct = 0
    embedding_list = []
    # [use_embedder.embed(splits[0]) for _ in range(5)]
    # use_embedder.embed_session.close()
    
    for s in splits:
        ct += 1
        print('-- batch', ct)
        use_embedder.embed(s)
        # DEBUG: somehow memory is running out
        # embedding_list.append(use_embedder.embed(s))
    embedding = np.array(embedding_list)
    embedding_reshaped = np.reshape(embedding, shape + (embedding.shape[-1],))
    return embedding_reshaped


def prepare_data_using_use(articles, summaries, scores):
    """Return vector of float32. The dimension of X is (?,13,512), Y is
    (?) where ? is the number of articles.
    """
    ARTICLE_MAX_SENT = 10
    SUMMARY_MAX_SENT = 3
    # (#num, 10, 512)
    print('creating sentence embedder instance ..')
    use_embedder = SentenceEmbedder()
    print('sentence embedding articles ..')
    article_data = sent_embed_articles(articles, ARTICLE_MAX_SENT, use_embedder)
    # (#num, 3, 512)
    # len(articles)
    article_data.shape
    # len(summaries)
    print('sentence embedding summaries ..')
    summary_data = sent_embed_articles(summaries, SUMMARY_MAX_SENT, use_embedder)
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
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    return (x_train, y_train), (x_val, y_val)



def split_sent_and_pad(articles, maxlen):
    res = []
    for article in articles:
        sents = sentence_split(article)
        sents_data = pad_sequences([sents], value='', padding='post',
                                   dtype=object, maxlen=maxlen)[0]
        # the shape is (#sent, 512)
        res.append(sents_data)
    return np.array(res)
    

def prepare_data_string(articles, summaries, scores):
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
