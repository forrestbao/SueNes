import os
import pickle
import shutil

import numpy as np
import glob
import json
import math

from bs4 import BeautifulSoup

from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
from antirouge.utils import sentence_split
from antirouge.utils import dict_pickle_read, dict_pickle_read_keys, dict_pickle_write

from antirouge.embedding import sentence_embed, sentence_embed_reset

import random
import tensorflow as tf

from antirouge.preprocessing import get_art_abs, embed_keep_shape

# FIXME DEPRECATED
from antirouge.config import *
from antirouge import config

def serial_process_story(in_dir, out_dir, batch_size=1000):
    """Save as a list. [(article, summary) ...].

    in_dir CNN_TOKENIZED_DIR, should contains a list of xxx.story,
    text with both article and summary
    
    out_dir os.path.join(SERIAL_DIR, 'story'), will output 1.pkl
    """
    # 92,579 stories
    stories = os.listdir(in_dir)
    ct = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res = []
    fname_ct = 0
    for key in stories:
        ct += 1
        story_file = os.path.join(in_dir, key)
        article, summary = get_art_abs(story_file)
        res.append((key, article, summary))
        if ct % batch_size == 0:
            fname_ct += 1
            fname = os.path.join(out_dir, "%s.pkl" % fname_ct)
            print('writing %s stories to %s' % (ct, fname))
            with open(fname, 'wb') as f:
                pickle.dump(res, f)
            res = []

def glob_sorted(pattern):
    """Sort according to the number in the filename."""
    return sorted(glob.glob(pattern), key=lambda f:
                  int(''.join(filter(str.isdigit, f))))

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

def serial_process_embed(base_folder, embedder, reset_interval=None):
    """Process folder/story and write folder/XXX where XXX is the embedder
name.

    base_folder SERIAL_DIR

    """
    assert embedder in ['USE', 'USE-Large', 'InferSent']
    out_dir = os.path.join(base_folder, embedder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # for each file in story folder, generate the embedding
    story_folder = os.path.join(base_folder, 'story')
    all_files = glob_sorted(story_folder + '/*')
    ct = 0
    for fname in all_files:
        ct += 1
        out_fname = os.path.join(out_dir, os.path.basename(fname))
        if os.path.exists(out_fname):
            print('%s already exists. Skip.' % out_fname)
            continue
        with open(fname, 'rb') as f:
            stories = pickle.load(f)
            keys = [story[0] for story in stories]
            articles = [story[1] for story in stories]
            summaries = [story[2] for story in stories]
            print('articles: ', len(articles))
            print('split sentences ..')
            # sentence split
            articles = [sentence_split(a) for a in articles]
            summaries = [sentence_split(s) for s in summaries]
            assert len(articles) == len(summaries)
            # encode
            print('encoding sentences ..')
            embedded = embed_keep_shape(articles + summaries, embedder)
            articles = embedded[:len(articles)]
            summaries = embedded[len(articles):]
            # articles = embed_keep_shape(articles, embedder)
            # summaries = embed_keep_shape(summaries, embedder)
            # list(zip([1,2,3], [4,5,6], [7,8,9]))
            obj = list(zip(keys, articles, summaries))
            print('writing to %s ..' % out_fname)
            with open(out_fname, 'wb') as f:
                pickle.dump(obj, f)
            if reset_interval is not None and ct % reset_interval == 0:
                print('reseting tf graph ..')
                sentence_embed_reset()
                tf.reset_default_graph()

def _create_tokenizer_CNN():
    # read all stories
    story_folder = os.path.join(CNN_SERIAL_DIR, 'story')
    all_files = glob_sorted(story_folder + '/*')
    all_text = []
    for fname in all_files:
        with open(fname, 'rb') as f:
            stories = pickle.load(f)
            keys = [story[0] for story in stories]
            articles = [story[1] for story in stories]
            summaries = [story[2] for story in stories]
            all_text.extend(articles)
            all_text.extend(summaries)
    tokenizer = create_tokenizer_from_texts(all_text)
    fname = os.path.join(CNN_SERIAL_DIR, 'tokenizer.json')
    save_tokenizer(tokenizer, fname)
    
def load_tokenizer_CNN():
    fname = os.path.join(CNN_SERIAL_DIR, 'tokenizer.json')
    if not os.path.exists(fname):
        print('Tokenizer %s not exists. Creating ..' % fname)
        _create_tokenizer_CNN()
        print('Tokenizer created. Loading ..')
    return load_tokenizer(fname)

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
