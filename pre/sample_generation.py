# GNU GPL 3.0 
# Forrest Sheng Bao 2020

# this module contains two functions for generating labeled samples 
# cross_pair and mutate 
# and functions that they call

# functions with examples have been tested

import random
import itertools, re, os
import joblib
import copy

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
    s = s.lower()
    s = replace_special_character(s, special_chars)

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

        dataset = tfds.load(name=dataset_name, split=
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
                [_doc, _sum] = line[:-1].split("\t")
                pairs.append((_doc, _sum))

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
#    return zip([i]*num_samples, sum_indexes)
    return (i, sum_indexes)

def cross_pair(data_pairs, neg_pos_ratio, dump_to, 
               in_memory, n_jobs):
    """Create positive and negative samples using cross pairing 

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary)

        neg_pos_ratio: int, ratio of negative sampels vs positive samples 
                       should be >= 1  

        dump_to: str, file path to dump the labeled doc-sum pair

        in_memory: Bool, whether to return labeled samples in memory

        n_jobs: int, number of CPU cores to use

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

    print ("Start generating sample indexes")
    neg_sample_indexes = joblib.Parallel(n_jobs=n_jobs)\
                         (joblib.delayed (cross_index) \
                                 (num_pair, i, neg_pos_ratio) for i in range(num_pair))

    print("Done generating sample indexes")
    print ("Start popping samples")

    for (doc_index, neg_sum_indexes) in neg_sample_indexes:
        [_doc, pos_sum] = data_pairs[doc_index]
        line = [_doc, pos_sum, "1"]
        for neg_sum_index in neg_sum_indexes:
            neg_sum = data_pairs[neg_sum_index][1]
            if in_memory:
                samples.append((_doc, neg_sum, 0))
            line += [neg_sum, "0"]
        if dump_to != None:
            f.write("\t".join(line) + "\n")

    print ("Done popping samples from indexes ")

    if dump_to != None:
        f.close()
    
    return samples

### mutation and associated functions 


def build_vocab(data_pairs, sent_end):
    """Build a set of all vocabularies, tokenized by spaces, from pairs of documents and summaries  
    """

    long_doc = " ".join([_doc for _doc, _sum in data_pairs])
    long_sum = " ".join([_sum for _doc, _sum in data_pairs])
    all_sent = long_doc + " " + long_sum   + " "

    # print (all_sent)
    for s in sent_end:
        all_sent = all_sent.replace(s, " ")
    # all_sent = re.sub("|".join(map(auto_escape, sent_end)), " ", all_sent)
    # print (all_sent)
    vocab = set(all_sent.split(' '))

    vocab.remove("")

    return vocab 
# build_vocab([(" A B? ", " 1? 2 "), (" C. D", " 3 4! ")], [".", "!", "?"])
# build_vocab([("A B", "1. 2"), ("C. D", "3! 4?")], [".", "!", "?"])


def mutate_add(words, vocab, ratio, sent_end):
    """add _ratio_% of vocab into random locations in _words_

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, 
    ratio: float, 0 to 1. 
    sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]

    example: 
        >>> mutate_add("i am a happy guy now".split(' '), ["hhhhh","jjjjj"], 0.2)
        'i am a happy jjjjj guy now'

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
        >>> mutate_delete("i am a happy guy now".split(' '), 0.2, ["."])
        'am a guy now'

    """
    words =  copy.deepcopy(words)
    length = len(words)
    indices = random.sample(range(length), int( ratio * length))

    return ' '.join([words[i] for i in range(length) \
                    if i not in indices or words[i][-1] in sent_end])

def mutate_replace(words, vocab, ratio, sent_end):
    """replace _ratio_% of words in random locations of _words_, 
    while preserving sentence separators

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, 
    ratio: float, 0 to 1  
    sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]


    example:
        >>> mutate_replace("i am a happy guy now".split(' '), ["hhhhh", "jjjjjjj"], 0.2, ["."]
    ...: )
        'i am a jjjjjjj guy now'

    """
    words =  copy.deepcopy(words)

    length = len(words)
    indices = random.sample(range(length), int(ratio * length))
    for i in indices: 
        if words[i][-1] not in sent_end:
            words[i] = vocab[random.randrange(len(vocab))]
    return ' '.join(words)

def mutate_switch(pair, all_vocab, method, ratios, sent_end):
    """Switch between 3 mutation methods,
    given a pair of document and summary, a list of vocabulary, 
    and a list of ratios.
    """
    
    (_doc, _sum)  = pair 
    # split the words and then feed to mutator 
    splitted_summary = _sum.split(' ')
    mutated = [] 

    for ratio in ratios: 
        # print (splitted_summary, end=" ")
        # print ("->", end=" ")

        if method == "add":
            mutated_tmp = mutate_add(splitted_summary, all_vocab, ratio, sent_end)
        elif  method == "delete":
            mutated_tmp = mutate_delete(splitted_summary, ratio, sent_end)
        elif  method == "replace":
            mutated_tmp = mutate_replace(splitted_summary, all_vocab, ratio, sent_end)
        else: 
            mutated_tmp = None 

        mutated.append((mutated_tmp, ratio))

        # print (mutated_tmp)

    return (_doc, mutated)

def mutate(data_pairs, ratios, method, sent_end, dump_to, n_jobs):
    """Create positive and negative samples using cross pairing 

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary) 

        ratios: list of floats, each of which is 0 to 1. The ratio of mutation. 

        method: str, one of ["add", "delete", "replace"]

        sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]

        dump_to: str, file path to dump the labeled doc-sum pair

        n_jobs: int, number of CPU cores to use

    return: list of triplets, [doc, mutated sum, ratio]
         
    example: 
         >>> list(mutate([("A B", "1. 2"), ("C. D", "3! 4?")], [0, 0.5, 1], 'replace', ['.', '!'], None, 4 ))
            [('A B', [('1. 2', 0), ('1. 1', 0.5), ('1. 4?', 1)]), 
             ('C. D', [('3! 4?', 0), ('3! C', 0.5), ('3! B', 1)])]
            # Note that 4? is treated as one word because ? is not in sent_end

        >>> list(mutate([("A B", "1. 2"), ("C. D", "3! 4?")], [0, 0.5, 1], 'add', ['.', '!'], None, 4 ))
            [('A B', [('1. 2', 0), ('A 1. 2', 0.5), ('2 1. A 2', 1)]),
             ('C. D', [('3! 4?', 0), ('A 3! 4?', 0.5), ('B 3! 4? 4?', 1)])]

        >>> list(mutate([("A B", "1. 2"), ("C. D", "3! 4?")], [0, 0.5, 1], 'delete', ['.', '!'], None, 4 ))
            [('A B', [('1. 2', 0), ('1. 2', 0.5), ('1.', 1)]),
             ('C. D', [('3! 4?', 0), ('3!', 0.5), ('3!', 1)])]

    """
    all_vocab = list(build_vocab(data_pairs, sent_end))
    mutated = []

    # print (all_vocab)

    triples  = joblib.Parallel(n_jobs=n_jobs)\
                         (joblib.delayed (mutate_switch) \
                                 (pair, all_vocab, method, ratios, sent_end) for pair in data_pairs)

    mutated = itertools.chain(triples)

    if dump_to != None:
        with open(dump_to, 'w')  as f:
            for (_doc, mutate_tuples)  in mutated: 
                line = [_doc]
                for (mutated_tmp, ratio) in mutate_tuples:
                    line +=  [mutated_tmp, str(ratio)]
                f.write("\t".join(line))
                f.write("\n")

    return mutated 

### put everything together 懶人包

def sample_generation():
    """

    2020-2-4: tested using pairs injected again
    """

    import sample_conf as cfg
    dataset_name = cfg.dataset_name 

    for split in cfg.splits:
        pairs = load_pairs(cfg.dataset_name, split, cfg.load_percent, cfg.num_shards, cfg.features, cfg.special_characters_to_clean, cfg.load_from, cfg.scramble, cfg.save_tsv)
        # pairs = [("A B", "1. 2"), ("C D", "3? 4")] # for testing
        for method in cfg.methods: 
            print ("generating samples using {} from dataset {}'s {} set"\
                      .format(method, dataset_name, split))
            if method in ["add", "delete", "replace"]:
                # NOTE: vocabulary generation is repeated here
                samples = mutate(pairs, cfg.mutate_ratios, method, cfg.sent_end, 
                        eval(cfg.dump_to), cfg.n_jobs)
            elif method in ["cross"]:
                samples = cross_pair(pairs, cfg.neg_pos_ratio, 
                        eval(cfg.dump_to), cfg.in_memory, cfg.n_jobs)

    return samples # only the last one 


if __name__ == "__main__":
    _ = sample_generation()



