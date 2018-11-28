#!/usr/bin/env python3

import pickle
def save_data(x_train, y_train, x_val, y_val):
    data = [x_train, y_train, x_val, y_val]
    with open('data.json', 'w') as f:
        json.dump(f, data)
def load_data():
    with open('data.json') as f:
        x_train, y_train, x_val, y_val = json.load(f)
        return x_train, y_train, x_val, y_val

def sentence_split(s):
    # FIXME keep the seperator
    res = re.split(r'\.|!|\?', s)
    res = [r.strip() for r in res if r]
    return res

def test():
    test_str = 'hello hello hello . world world ! eh eh eh ? yes yes ... ok ok'
    sentence_split(test_str)

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
