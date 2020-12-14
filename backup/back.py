def test_random_word_generator(random_word_generator):
    """
    0.08432126045227051
    0.8023912906646729
    7.96087908744812
    ----
    1.4066696166992188e-05
    7.390975952148438e-05
    0.00061798095703125
    ----
    0.008105278015136719
    0.008072376251220703
    0.00867319107055664
    """
    t = time.time()
    _ = [random_word_generator.random_word() for _ in range(10)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word() for _ in range(100)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word() for _ in range(1000)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word_optimized() for _ in range(10)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word_optimized() for _ in range(100)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_word_optimized() for _ in range(1000)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_words(10)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_words(100)]
    print(time.time() - t)
    t = time.time()
    _ = [random_word_generator.random_words(1000)]
    print(time.time() - t)

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




def sentence_embed_old(embed_name, sentences, batch_size):
    embed_func = {'USE': _sentence_embed_USE_small,
                  'USE-Large': _sentence_embed_USE_large,
                  'InferSent': _sentence_embed_InferSent}[embed_name]

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    res = []
    print('embedding %s sentences, batch size: %s'
          % (len(sentences), batch_size))
    # rg = range(0, len(sentences), batch_size)

    # sort by length in descending order to avoid OOM
    length = np.array([-len(sentence) for sentence in sentences])
    index = np.argsort(length)

    msg = ''
    start = time.time()
    limit = 10000
    batch = []
    enum = range(len(index))
    for i in enum:
        batch.append(sentences[index[i]])
        if -length[index[i]] >= limit or len(batch) == batch_size or i == len(index)-1:
            print('\b' * len(msg), end='')
            total_time = time.time() - start
            if i == 0:
                eta = -1
            else:
                eta = (total_time / i) * (len(index) - i)
            speed = 0 if total_time == 0 else i / total_time
            msg = ('batch size: %s, iteration num %s / %s, '
                'speed: %.0f sent/s, Total Time: %.0fs, ETA: %.0fs'
                % (batch_size, i, len(index), speed, total_time, eta))
            print(msg, end='', flush=True)

            tmp = np.array(embed_func(batch))
            res.append(tmp)
            batch = []
    
    print('')
    # reorder
    tmp = np.vstack(res)
    result = [None] * len(index)
    for i in enum:
        result[index[i]] = tmp[i]
    
    return np.array(result)

    '''
    for idx,stidx in enumerate(rg):
        # I have to set the batch size really small to avoid
        # memory or assertion issue. Thus there will be many batch
        # iterations. The update of python shell buffer in Emacs
        # is very slow, thus only update this every severl
        # iterations.
        # print('\r')
        if embed_name != 'InferSent' or idx % 30 == 0:
            print('\b' * len(msg), end='')
            total_time = time.time() - start
            if idx == 0:
                eta = -1
            else:
                eta = (total_time / idx) * (len(rg) - idx)
            speed = 0 if total_time == 0 else batch_size * idx / total_time
            msg = ('batch size: %s, batch num %s / %s, '
                   'speed: %.0f sent/s, Total Time: %.0fs, ETA: %.0fs'
                   % (batch_size, idx, len(rg), speed, total_time, eta))
            print(msg, end='', flush=True)
        batch = sentences[stidx:stidx + batch_size]
        tmp = embed_func(batch)
        res.append(tmp)
    print('')
    return np.vstack(res)
    '''



_InferSent_model = None
def _sentence_embed_InferSent(sentences):
    global _InferSent_model
    if not _InferSent_model:
        # Load our pre-trained model (in encoder/):
        # this bsize seems not used at all
        params_model = {'bsize': 128,
                        'word_emb_dim': 300,
                        'enc_lstm_dim': 2048, 'pool_type': 'max',
                        'dpout_model': 0.0,
                        # must use the v2 model in INFERSENT_MODEL_PATH
                        'version': 2}
        _InferSent_model = InferSent(params_model)
        _InferSent_model.cuda()
        _InferSent_model.load_state_dict(torch.load(get_infersent_modelpath()))
        _InferSent_model.set_w2v_path(get_infersent_w2vpath())
        _InferSent_model.build_vocab_k_words(K=100000)
    embeddings = _InferSent_model.encode(sentences, bsize=128,
                                         tokenize=False, verbose=False)
    # This outputs a numpy array with n vectors of dimension
    # 4096. Speed is around 1000 sentences per second with batch
    # size 128 on a single GPU.
    return embeddings

def sentence_embed_reset():
    global _USE_small_module
    global _USE_small_sess
    global _USE_large_module
    global _USE_large_sess
    if _USE_small_module:
        _USE_small_module = None
        _USE_small_sess.close()
        _USE_small_sess = None
    if _USE_large_module:
        _USE_large_module = None
        _USE_large_sess.close()
        _USE_large_sess = None
    



_USE_small_module = None
_USE_small_sess = None
def _sentence_embed_USE_small_tf1(sentences):
    global _USE_small_module
    global _USE_small_sess
    device = '/cpu:0'
    url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    if not _USE_small_module:
        _USE_small_module = hub.Module(url)
        config = tf.ConfigProto(allow_soft_placement = True)
        # even this does not use GPU, it will still take all GPU
        # memories
        _USE_small_sess = tf.Session(config = config)
        _USE_small_sess.run(tf.global_variables_initializer())
        _USE_small_sess.run(tf.tables_initializer())
    with tf.device(device):
        return _USE_small_sess.run(_USE_small_module(sentences))

def _sentence_embed_USE_small_tf2(sentences):
    global _USE_small_module
    url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    if not _USE_small_module:
        _USE_small_module = hub.load(url)
    return _USE_small_module(sentences)

_sentence_embed_USE_small = _sentence_embed_USE_small_tf2 if tf.__version__.startswith('2') else _sentence_embed_USE_small_tf1

_USE_large_module = None
_USE_large_sess = None
def _sentence_embed_USE_large_tf1(sentences):
    global _USE_large_module
    global _USE_large_sess
    device = '/gpu:0'
    # CPU is too slow for this model
    # device = '/cpu:0'
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    if not _USE_large_module:
        _USE_large_module = hub.Module(url)
        config = tf.ConfigProto(allow_soft_placement = True)
        # this does not seem to have any effect
        # config.gpu_options.allow_growth = True
        _USE_large_sess = tf.Session(config = config)
        # _USE_large_sess = tf.Session()
        _USE_large_sess.run(tf.global_variables_initializer())
        _USE_large_sess.run(tf.tables_initializer())
    with tf.device(device):
        return _USE_large_sess.run(_USE_large_module(sentences))

def _sentence_embed_USE_large_tf2(sentences):
    # Memory leak for unknown reason?
    global _USE_large_module
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    if not _USE_large_module:
        _USE_large_module = hub.load(url)
    return _USE_large_module(sentences)

_sentence_embed_USE_large = _sentence_embed_USE_large_tf2 if tf.__version__.startswith('2') else _sentence_embed_USE_large_tf1



def __test_nce_loss():
    """Create a model to test NCE loss.

    1. is the labels always 1?

    The plan: y = 1 if 0<x<100 else 0
    
    """
    # generate data
    # how about using mnist data?
    
    pass



def __test_keras_preprocessing():
    tokenizer.texts_to_sequences(['hello world you are awesome', 'you is good person'])
    tokenizer.sequences_to_texts([[10468, 88, 35, 22, 6270], [35, 11, 199, 363]])
    one_hot('hello world you are you awesome', 200)
    return



def test_infersent():
    sentences = ['Everyone really likes the newest benefits ',
                 'The Government Executive articles housed on the website are not able to be searched . ',
                 'I like him for the most part , but would still enjoy seeing someone beat him . ',
                 'My favorite restaurants are always at least a hundred miles away from my house . ',
                 'I know exactly . ',
                 'We have plenty of space in the landfill . '] * 10000


    embedder = InferSentEmbedder(bsize=1024)
    embeddings = embedder.embed(sentences)
    embeddings
    embeddings.shape
    # import pickle
    # pickle.load(open('encoder/infersent2.pkl', 'rb'))
    # import nltk
    # nltk.download('punkt')


def test_USE():
    sentences = ["The quick brown fox jumps over the lazy dog."]
    sentences = ['Everyone really likes the newest benefits ',
                 'The Government Executive articles housed on the website are not able to be searched . ',
                 'I like him for the most part , but would still enjoy seeing someone beat him . ',
                 'My favorite restaurants are always at least a hundred miles away from my house . ',
                 'I know exactly . ',
                 'We have plenty of space in the landfill . '] * 10000
    embedder = UseEmbedder(encoder='transformer', bsize=10240, gpu=True)
    embedder = UseEmbedder(encoder='dan', bsize=10240, gpu=False)
    embeddings = embedder.embed(sentences)
    embeddings.shape
    embeddings

def __test():
    text_dir = os.path.join(DUC_2002_DIR, 'text')
    embed_dir = os.path.join(DUC_2002_DIR, 'text_embedded')
    batch_size = 10000
    embed_folder(text_dir, embed_dir, 'USE', batch_size)
    embed_folder(text_dir, embed_dir, 'USE-Large', batch_size)
    embed_folder(text_dir, embed_dir, 'InferSent', batch_size)

def __test_embed_result():
    def read_embed(ID):
        with open(os.path.join(DUC_2002_DIR, 'text_embedded/USE',
                               ID + '.pkl'), 'rb') as fp:
            return pickle.load(fp)
    def embed_onthefly(ID):
        with open(os.path.join(DUC_2002_DIR, 'text',
                               ID + '.txt')) as fp:
            text = fp.read()
            sents = [s for s in text.split('\n') if s]
            code = sentence_embed('USE', sents, USE_BATCH_SIZE)
            return code
    ID = 'AP891216-0037'
    ID = 'AP891216-0037--H'
    ID = 'AP900130-0010--30'
    ID = 'AP891118-0136--27'

    with open(os.path.join(config.DUC_2002_DIR, 'meta.json')) as fp:
        duc_meta = json.load(fp)
    doc_ids = [d[1] for d in duc_meta]
    doc_ids = list(set(doc_ids))
    for ID in random.sample(doc_ids, 5):
        print(ID)
        code1 = read_embed(ID)
        code2 = embed_onthefly(ID)
        print(((code1 - code2)**2).mean())




def embed_folder(in_dir, out_dir, embedder, batch_size):
    """Embed the text in the file and save the pickle file into the
out_file.

    The in_fname is assumed to have sentences seperated by
    newline. The output will be a numpy array with shape (#sent,
    512/4096)

    """
    out_dir = os.path.join(out_dir, embedder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    all_files = glob.glob(in_dir + '/*')
    # apply batch here
    num_batch = math.ceil(len(all_files) / batch_size)
    for i in range(num_batch):
        print('file batch (size: %s): %s / %s' % (batch_size, i, num_batch))
        files = all_files[i*batch_size:(i+1)*batch_size]
        # read all of them
        to_encode = []
        for _, fname in enumerate(files):
            with open(fname) as fp:
                content = fp.read()
                sents = [s for s in content.split('\n') if s]
                to_encode.append(sents)
        embedded = embed_keep_shape(to_encode, embedder)
        assert(len(embedded) == len(to_encode))
        for j, fname in enumerate(files):
            base = os.path.basename(os.path.splitext(fname)[0])
            out_fname = os.path.join(out_dir, base + '.pkl')
            with open(out_fname, 'wb') as fp:
                pickle.dump(embedded[j], fp)


def _create_tokenizer_DUC():
    # read all DUC text
    # create tokenizer based on them
    with open(os.path.join(config.DUC_2002_DIR, 'meta.json')) as fp:
        duc_meta = json.load(fp)
    doc_ids = [d[0] for d in duc_meta]
    abs_ids = [d[1] for d in duc_meta]
    ids = list(set(doc_ids)) + list(set(abs_ids))
    all_text = []
    for ID in ids:
        fname = os.path.join(config.DUC_2002_DIR, 'text', docID + '.txt')
        with open(fname) as fp:
            text = fp.read()
            all_text.extend(text)
    tokenizer = create_tokenizer_from_texts(all_text)
    fname = os.path.join(config.DUC_2002_DIR, 'tokenizer.json')
    save_tokenizer(tokenizer, fname)

def load_tokenizer_DUC():
    fname = os.path.join(config.DUC_2002_DIR, 'tokenizer.json')
    if not os.path.exists(fname):
        print('Tokenizer %s not exists. Creating ..' % fname)
        _create_tokenizer_DUC()
        print('Tokenizer created. Loading ..')
    return load_tokenizer(fname)

def process_DUC_RAW():
    """My objective is to get the article, summary, score.

    @return (docID, absID, score, doc_fname, abs_fname)
    """
    peer_dir = os.path.join(DUC_2002_RAW_DIR,
                            'results/abstracts/phase1/SEEpeers',
                            'SEE.abstracts.in.sentences')
    peer_baseline_dir = os.path.join(DUC_2002_RAW_DIR,
                                     'results/abstracts/phase1/SEEpeers',
                                     'SEE.baseline1.in.sentences')
    result_file = os.path.join(DUC_2002_RAW_DIR,
                               'results/abstracts/phase1/short.results.table')
    doc_dir = os.path.join(DUC_2002_RAW_DIR, 'data/test/docs.with.sentence.breaks')
    # get the list of (docID, absID, score)
    res = []
    with open(result_file) as f:
        for line in f:
            if line.startswith('D'):
                splits = line.split()
                if splits[1] == 'P':
                    docsetID = splits[0]
                    docID = splits[2]  # ***
                    length = splits[3]  # seems that all lengths are 100
                    selector = splits[5]
                    # summarizer = splits[6]
                    # assessor = splits[7]
                    absID = splits[8]  # ***
                    score = splits[27]     # ***
                    # docset.type.length.[selector].peer-summarizer.docref
                    fname = '%s.%s.%s.%s.%s.%s.html' % (docsetID, 'P',
                                                        length,
                                                        selector,
                                                        absID,
                                                        docID)
                    doc_fname = os.path.join(doc_dir,
                                             docsetID.lower()+selector.lower(),
                                             docID+'.S')
                    if absID == '1':
                        abs_fname = os.path.join(peer_baseline_dir, fname)
                    else:
                        abs_fname = os.path.join(peer_dir, fname)
                    if not os.path.exists(doc_fname):
                        print('File not found: ', doc_fname)
                    elif not os.path.exists(abs_fname):
                        print('File not found: ', abs_fname)
                    else:
                        # absID is augmented with docID
                        res.append((docID, docID + '--' + absID, float(score),
                                    doc_fname, abs_fname))
    return res

# doc file example: /home/hebi/mnt/data/nlp/DUC2002/data/test/docs.with.sentence.breaks/d061j/AP880911-0016.S
# abs file example: /home/hebi/mnt/data/nlp/DUC2002/results/abstracts/phase1/SEEpeers/SEE.baseline1.in.sentences/D061.P.100.J.1.AP880911-0016.html
def _copy_duc_text_impl(in_fname, out_fname, doc_type):
    """Parse the content of in file and output to out file."""
    # TODO should I use a rigrious html parser instead of bs4?
    assert(doc_type in ['doc', 'abs'])
    print('copying ', in_fname, 'into', out_fname)
    selector = 'TEXT s' if doc_type == 'doc' else 'body a[href]'
    with open(in_fname) as fin, open(out_fname, 'w') as fout:
        soup = BeautifulSoup(fin)
        for s in soup.select(selector):
            sent = s.get_text()
            fout.write(sent)
            fout.write('\n\n')

def copy_duc_text(duc_meta, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for d in duc_meta:
        docID = d[0]
        # absID = d[0] + '--' + d[1]
        doc_fname = d[3]
        abs_fname = d[4]
        doc_out = os.path.join(out_dir, docID + '.txt')
        abs_out = os.path.join(out_dir, absID + '.txt')
        if not os.path.exists(doc_out):
            _copy_duc_text_impl(doc_fname, doc_out, 'doc')
        if not os.path.exists(abs_out):
            _copy_duc_text_impl(abs_fname, abs_out, 'abs')

def process_DUC():
    # get meta data
    duc_meta = process_DUC_RAW()
    duc_meta[0]

    # copy text to folder
    len(duc_meta)
    copy_duc_text(duc_meta, os.path.join(DUC_2002_DIR, 'text'))

    duc_meta_simple = [d[:3] for d in duc_meta]
    with open(os.path.join(DUC_2002_DIR, 'meta.json'),'w') as fp:
        json.dump(duc_meta_simple, fp, indent=4)
# DEPRECATED
def build_glove_summary_only_model(embedding_layer):
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(config.SUMMARY_MAX_WORD,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model
