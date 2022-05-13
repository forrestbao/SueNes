# GNU GPL 3.0 
# Forrest Sheng Bao 2020

# this module contains two functions for generating labeled samples 
# cross_pair and mutate 
# and functions that they call

# functions with examples have been tested

import random
import itertools, re, os
import joblib, multiprocessing
import copy
import functools
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

#===== lexical processing

def auto_escape(s):
    """Return the escape sequence of a character if it is special
    """
    d = {"?": "\\?", ".": "\\."}
    try:
        return d[s]
    except KeyError:
        return s
auto_escape(".")


def replace_special_character(s, L):
    """replaces all special characters in _L_ within _s_ by spaces
    """
    for l in L:
        s= s.replace(l, " ")
    return s

def normalize_sentence(s, special_chars):
    """Normalizing a sentence
    """
    # s = s.lower()
    s = replace_special_character(s, special_chars)
    s = s[:4000] # up to 500 characters per article or summary, useful when article is very long.

    return s 

#====== data loading

def load_pairs(dataset_name, split, load_percent, num_shards, 
               features, special_chars, load_from, scramble, save_tsv):
    """Load pairs of documents and their summaries

    dataset_name: str, a unique name to ref to it in tfds 
        e.g., 'cnn_dailymail', 'newsroom', 'big_patent'

    split: str, 'test', 'train' or 'validation' 
           results in a _OptionsDataset type 

    load_percent: int, 0 to 100, ratio of data to take from the dataset or data split. 
                100 means use all data

    num_shards: int, creates a Dataset that includes only 1/num_shards of this dataset,
                to be used in downstreams

    features: [str, str], names of the document and the summary in TFDS, 
              e.g., ['article', 'highlights'] in CNN_DailyMail

    special_chars: list of strings, special characters to be replaces by spaces, 
                   e.g., ['\t', '\n']

    load_from: str, load data from tfds or tsv. Default TFDS. 
    
    scramble: bool, whether to sramble the loaded data
                    only effective when load_from is TFDS.

    save_tsv: bool, whether to save doc-sum pairs into a TSV file 
                    for computers without TF2 or TFDS. 

    """

    tsv_filename = "./" + dataset_name + "_" +split + "_" + \
                   str(load_percent) + "_" + str(num_shards) + ".tsv"

    if load_from == "tfds":

        import tensorflow_datasets as tfds 
        print ("Loading data. If the data not available locally, download first.")

        dataset = tfds.load(name=dataset_name, download=True, split=
                split+ '[{}%:{}%]'.format(0, load_percent)
                )

        if scramble: 
            dataset.shuffle(4096)

        dataset = dataset.shard(num_shards=num_shards, index=0)

    #    plain_pairs = [(piece[features[0]], piece[features[1]]) for piece in dataset]

        pairs = [(normalize_sentence(piece[features[0]].numpy().decode("utf-8"), special_chars), 
                  normalize_sentence(piece[features[1]].numpy().decode("utf-8"), special_chars) )
                  for piece in dataset]

        if save_tsv and load_from == "tfds":
            with open(tsv_filename, 'w') as f:
                for (_doc, _sum) in pairs:
                    f.write("\t".join([_doc, _sum]))
                    f.write("\n")
                                   
    elif load_from == "tsv":
        pairs = []
        with open(tsv_filename, 'r') as f:
            for line in f:
                try:
                    [_doc, _sum] = line.replace("\n","").split("\t")
                    pairs.append((_doc, _sum))
                except ValueError:
                    print ("skipping this line:", line)

    return pairs 

#========== crosspairing and associated functions 

def cross_index(n,i,r):
    """Given an index _i_, and a range from 0 to _n_, randomly sample 
    _r_ or (_n_-1) numbers, whatever is smaller, from 0 to n 
    such that they do not equal to _i_. 

    >>> list(cross_index(5,1,3))
        (1, [4, 2, 0])


    """
    index_pool = list(itertools.chain(range(0,i), range(i+1,n)))
    num_samples = min(n-1, r)
    sum_indexes = random.sample(index_pool, num_samples)
    return (i, sum_indexes)

def cross_pair(data_pairs, neg_pos_ratio, dump_to, 
               in_memory, n_jobs, dump_format):
    """Create positive and negative samples using cross pairing 

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary)

        neg_pos_ratio: int, ratio of negative sampels vs positive samples 
                       should be >= 1  

        dump_to: str, file path to dump the labeled doc-sum pair

        in_memory: Bool, whether to return labeled samples in memory

        n_jobs: int, number of CPU cores to use

        dump_format: str, "compact" or "plain". The dump format, see conf. 

    return: list of triplets, [doc, sum, 0 or 1]
            0 if sum and doc do not match. 1 if so. 

    example: 
        >>> sample_generation.cross_pair([("A", "1"),("B", "2"), ("C", "3")], 1, None, True, 4)
            [('A', '1', 1),
             ('B', '2', 1),
             ('C', '3', 1),
             ('A', '2', 0),
             ('B', '1', 0),
             ('C', '1', 0)]

        >>> sample_generation.cross_pair([("A", "1"),("B", "2"), ("C", "3")], 20, None, True, 16)
            [('A', '1', 1),
             ('B', '2', 1),
             ('C', '3', 1),
             ('A', '3', 0),
             ('A', '2', 0),
             ('B', '3', 0),
             ('B', '1', 0),
             ('C', '2', 0),
             ('C', '1', 0)]

    """
    samples = []

    # positive samples 
    if dump_to != None:
        f = open(dump_to, 'w')

    if in_memory:
        samples = [(_doc, _sum, 1) for (_doc,_sum) in data_pairs]

    # negative samples
    num_pair = len(data_pairs)

    print ("\t Generating random sample indexes...", end=" ")
    neg_sample_indexes = joblib.Parallel(n_jobs=n_jobs)\
                         (joblib.delayed (cross_index) \
                                 (num_pair, i, neg_pos_ratio) for i in range(num_pair))

    print("\tDone ")
    print ("\tPopping samples from random indexes", end=" ")

    for (doc_index, neg_sum_indexes) in neg_sample_indexes:
        [_doc, pos_sum] = data_pairs[doc_index]
        line = [_doc, pos_sum, "1"]
        if dump_to != None and dump_format == "plain": 
            f.write("\t".join(line) + "\n")

        for neg_sum_index in neg_sum_indexes:
            neg_sum = data_pairs[neg_sum_index][1]
            if in_memory:
                samples.append((_doc, neg_sum, 0))
            
            if dump_to != None:
                if dump_format == "plain":
                    line = [_doc, neg_sum, "0"]
                    f.write("\t".join(line) + "\n")
                elif dump_format == "compact":
                    line += [neg_sum, "0"]

        if dump_to != None and dump_format == "compact":
            f.write("\t".join(line) + "\n")

    print ("Done ")

    if dump_to != None:
        f.close()
    
    return samples

### mutation and associated functions 

def pair2words(data_pair, sent_end):
    """Extract all words in one pair of document and summary 

    Examples: 
    >>> pair2words((" A B. ", "1. 2 "), [".", "!"])
        ['', 'A', 'B', '', '1', '', '2', '']
    >>> pair2words((" A B. ", "1. 2 "), ["?", "!"])
        ['', 'A', 'B.', '1.', '2', '']
    >>> pair2words((" A   B. ", "1. 2 "), ["?", "!"])
        ['', 'A', '', '', 'B.', '1.', '2', '']
    """
    doc_and_sum = data_pair[0] + " " +data_pair[1]

    for e in sent_end:
        doc_and_sum = doc_and_sum.replace(e, " ")

    return doc_and_sum.split()

def build_vocab(data_pairs, sent_end, n_jobs):
    """Build a set of all vocabularies, tokenized by spaces, from pairs of documents and summaries  
    """

    print ("\t Building vocabulary..." , end  = " ")

    with multiprocessing.Pool(n_jobs) as p:
        words = p.map(functools.partial(pair2words, sent_end=sent_end), data_pairs)
    
    print (" Word list generated", end="...")
    words = itertools.chain(*words)
    # this step takes too much memory, even with single core 
    vocab = set(words)

    print ("Vocabulary built")

    if "" in vocab:
        vocab.remove("")

    return vocab 
# build_vocab([(" A B? ", " 1? 2 "), (" C. D", " 3 4! ")], [".", "!", "?"])
# build_vocab([("A B", "1. 2"), ("C. D", "3! 4?")], [".", "!", "?"])

def pick_random_word(data_pairs, sent_end):
    """Pick a random word from pairs of documents and summaries

    data_pairs: list of 2-tuples, [(doc1, sum1), (doc2, sum2), etc. ]
    sent_end: tuple of str, end of sentences 
    """
    upper = len(data_pairs)
    pair_index = random.randrange(upper)
    all_sentences = data_pairs[pair_index][0] + " " + data_pairs[pair_index][1]
    for e in sent_end:
        all_sentences = all_sentences.replace(e, " ")
    words = all_sentences.split()
    # print (all_sentences)
    # print (words)
    return random.choice(words)
    

def mutate_add(words, vocab, ratio, sent_end):
    """add _ratio_% of vocab into random locations in _words_

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, the vocabulary of all docs and summaries in a dataset
    ratio: float, 0 to 1. 
    sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]
    data_pairs: list of 2-tuples, [(doc1, sum1), (doc2, sum2), etc. ]

    example: 
        >>> mutate_add("i am a happy guy now".split(" "), [], 0.2,\
            string.punctuation, [("A B", "1. 2"), ("C. D", "3! 4?")])            
            'i am a happy A guy now'
    """
    words =  copy.deepcopy(words)
    length = len(words)
    indices = random.sample(range(length), int(ratio * length))
    res = []
    for i in range(length):
        if i in indices:
            candidate = vocab[random.randrange(len(vocab))] 
            res.append(candidate)
        res.append(words[i])
    return ' '.join(res)

def mutate_delete(words, ratio, sent_end):
    """delete _ratio_% of words in random locations of _words_, 
    while preserving sentence separators

    words: list of strings, e.g., ["I", "am", "lucky"]
    ratio: float, 0 to 1 
    sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]

    example: 
        >>> mutate_delete("i am a happy guy now".split(" "), 0.2, string.punctuation)
            'i am a guy'
    """
    words =  copy.deepcopy(words)
    length = len(words)
    indices = random.sample(range(length), int( ratio * length))

    try : 
        res=  ' '.join([words[i] for i in range(length) \
                    if i not in indices or words[i][-1] in sent_end])
    except IndexError:
        print (words )

    return res 

def mutate_replace(words, vocab, ratio, sent_end):
    """replace _ratio_% of words in random locations of _words_, 
    while preserving sentence separators

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, the vocabulary of all docs and summaries in a dataset
    ratio: float, 0 to 1  
    sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]


    example:
        >>> mutate_replace("i am a happy guy now".split(" "), [], \
            0.2, string.punctuation, [("A B", "1. 2"), ("C. D", "3! 4?")])
            'i am a B guy now'
    """
    words =  copy.deepcopy(words)

    length = len(words)
    try: 
        indices = random.sample(range(length), int(ratio * length))
    except ValueError:
        print (ratio, words)
        print ("Ratio error. quit. ")
        exit()
    for i in indices: 
        #with vocab 
        if words[i][-1] in sent_end:
            words[i] = vocab[random.randrange(len(vocab))] + words[i][-1]
        else: 
            words[i] = vocab[random.randrange(len(vocab))]
    return ' '.join(words)

# def mutate_switch(pair, vocab, method, neg_pos_ratio, sent_end, data_pairs):
def mutate_switch(pair, vocab, method, neg_pos_ratio, sent_end):
    """Switch between 3 mutation methods,
    given a pair of document and summary, a list of vocabulary, 
    and neg_pos_ratio. 

    """
    
    (_doc, _sum) = pair 
    # split the words and then feed to mutator 
#    print (_sum)
    splitted_summary = _sum.split()

    ratios = [random.uniform(0, 1) for _ in range(neg_pos_ratio)]
    ratios.sort() # This is the percentage of mutation. But scores will be 1-ratios. 
    # Hence acending order of ratio is Descending order of quality 
    # do NOT use reverse=True here

    mutated = [(_sum, 0.0)]  # 0.0 is the ratio. Later at 1-ratio, it flips to 1, meaning best quality
    for ratio in ratios: 
        # print (splitted_summary, end=" ")
        # print ("->", end=" ")

        if method == "word_add":
            mutated_tmp = mutate_add(splitted_summary, vocab, ratio, sent_end)
        elif  method == "word_delete":
            mutated_tmp = mutate_delete(splitted_summary, ratio, sent_end)
        elif  method == "word_replace":
            mutated_tmp = mutate_replace(splitted_summary, vocab, ratio, sent_end)
        else: 
            mutated_tmp = None 

        mutated.append((mutated_tmp, ratio))

        # print (mutated_tmp)

    return (_doc, mutated)

def mutate(data_pairs, neg_pos_ratio, method, sent_end, dump_to, n_jobs, dump_format):
    """Create positive and negative samples using one of the 3 mutation methods

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary) 

        neg_pos_ratio: int, ratio of negative sampels vs positive samples 
                should be >= 1  

        method: str, one of ["word_add", "word_delete", "word_replace"]

        sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]

        dump_to: str, file path to dump the labeled doc-sum pair

        n_jobs: int, number of CPU cores to use

        dump_format: str, "compact" or "plain". See cfg. 

    return: list of triplets, [doc, mutated sum, ratio]
         
    example: 
        >>> list(sample_generation.mutate([("A B", "1. 2"), ("C. D", "3! 4?")], 2, 'replace', ['.', '!', '?'], None, 4 , "plain"))
            [('A B', [('D. 2', 0.9743772514534714), ('1. 2', 0.01600339659373673)]),
             ('C. D', [('1! 4?', 0.8081154190302389), ('D! 4?', 0.5093176309531442)])]
            # Note that 4? is treated as one word because ? is not in sent_end
            # Since we do not fix random state, the result may vary. 
    """

    if method == "word_delete":
        vocab= []
    else:
        vocab = build_vocab(data_pairs, sent_end, n_jobs)
        vocab = list(vocab)

    mutated = []

    print ("\t Mutating in ", method, end = "... ")

    if dump_to != None:
        f = open(dump_to, 'w')

    # no parallelization 
    # mutated = [mutate_switch(pair, vocab, method, neg_pos_ratio, sent_end, data_pairs)
    #             for pair in data_pairs]

    # parallel with joblib
    # joblib sucks. do not use. 
    # mutated = joblib.Parallel(n_jobs=n_jobs)\
    #           (joblib.delayed 
    #           (functools.partial(mutate_switch, vocab=vocab, method=method, \
    #               neg_pos_ratio=neg_pos_ratio, sent_end = sent_end, data_pairs=data_pairs)) \
    #                         (pair) for pair in  data_pairs)

    # parallel with multiprocessing 
    # without vocab 
    # with multiprocessing.Pool(n_jobs) as p:
    #     mutated = p.map(functools.partial(mutate_switch, \
    #                     vocab=vocab, method=method, neg_pos_ratio=neg_pos_ratio, sent_end = sent_end,\
    #                     data_pairs=data_pairs), data_pairs)
    # with vocab 
    with multiprocessing.Pool(n_jobs) as p:
        mutated = p.map(functools.partial(mutate_switch, \
                        vocab=vocab, method=method, neg_pos_ratio=neg_pos_ratio, sent_end = sent_end), data_pairs)

    print ("Done" )

    print ("\t Writing mutation to file...", end  = " ")

    if dump_to != None:
        if dump_format == "compact":
            for (_doc, mutate_tuples) in mutated: 
                line = [_doc]
                for (mutated_tmp, ratio) in mutate_tuples:
                    line +=  [mutated_tmp, str(1-ratio)]
                f.write("\t".join(line))
                f.write("\n")

        elif dump_format == "plain":
            for (_doc, mutate_tuples) in mutated: 
                for (mutated_tmp, ratio) in mutate_tuples:
                    line =  [_doc, mutated_tmp, str(1-ratio)]
                    f.write("\t".join(line))
                    f.write("\n")

        f.close()
    print ("Done ")
    return mutated 

### put everything together 懶人包

def sample_generation(conf):
    """

    conf: str, a python module name 
    """
    samples = []
    cfg = __import__(conf)
    dataset_name = cfg.dataset_name 
    print (f'Generating from {dataset_name}')

    for split in cfg.splits:
        pairs = load_pairs(cfg.dataset_name, split, cfg.load_percent,\
                cfg.num_shards, cfg.features, cfg.special_characters_to_clean, \
                cfg.load_from, cfg.scramble, cfg.save_tsv)
        # pairs = [("A B", "1. 2"), ("C D", "3? 4")] # for testing
        for method in cfg.methods: 

            filename = eval(cfg.dump_to)

            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            print ("generating samples using {} from dataset {}'s {} set"\
                      .format(method, dataset_name, split))
            if method in ["word_add", "word_delete", "word_replace"]:
                # NOTE: vocabulary generation is repeated here
                samples = mutate(pairs, cfg.neg_pos_ratio, method, cfg.sent_end, 
                        filename, cfg.n_jobs, cfg.dump_format)
            elif method in ["cross"]:
                samples = cross_pair(pairs, cfg.neg_pos_ratio, 
                        filename, cfg.in_memory, cfg.n_jobs, cfg.dump_format)
            
            samples = [] # free space
            gc.collect() 

    return samples # only the last one 


if __name__ == "__main__":
    # Generate samples from CNN DM using configurations in cnndm_conf.py 
    _ = sample_generation("cnndm_conf")

    # Generate samples from Billsum
    _ = sample_generation("billsum_conf")

    # Generate samples from Scientific papers
    _ = sample_generation("scientific_papers_conf")

    # Generate samples from big patents
    _ = sample_generation("big_patents_conf")
