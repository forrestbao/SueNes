# GNU GPL 3.0 
# Forrest Sheng Bao 2020

# this module contains two functions for generating labeled samples 
# cross_pair and mutate 
# and functions that they call

# functions with examples have been tested


import tensorflow_datasets as tfds 

#===== lexical processing

def replace_special_character(s, L):
    """replaces all special characters in _L_ within _s_ by spaces
    """
    for l in L:
        s= s.replace(l, " ")
    return s


#====== data loading

def load_pairs(dataset_name, split, take_percent, features, special_chars ):
    """Load pairs of documents and their summaries

    dataset_name: str, a unique name to ref to it in tfds 
        e.g., 'cnn_dailymail', 'newsroom', 'big_patent'

    split: str, 'test', 'train' or 'validation' 
           results in a _OptionsDataset type 

    take_percent: int, 0 to 100, ratio of data to take from the dataset or data split. 100 means use all data 

    features: [str, str], names of the document and the summary in TFDS, e.g., ['article', 'highlights']

    special_chars: list of strings, special characters to be replaces by spaces, 
                   e.g., ['\t', '\n']

    """

    print ("Loading data. If the data not available locally, download first.")

    dataset = tfds.load(name=dataset_name, split=
            split+ '[{}%:{}%]'.format(0, take_percent)
            )

#    plain_pairs = [(piece[features[0]], piece[features[1]]) for piece in dataset]

    pairs = [(replace_special_character(piece[features[0]].numpy().decode("utf-8"), special_chars), 
              replace_special_character(piece[features[1]].numpy().decode("utf-8"), special_chars) )
             for piece in dataset]

    return pairs 

#========== crosspairing and associated functions 

def cross_pair(data_pairs, neg_pos_ratio, dump_to=None, in_memory=False ):
    """Create positive and negative samples using cross pairing 

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary)

        neg_pos_ratio: int, ratio of negative sampels vs positive samples 
                       should be >= 1  

        dump_to: str, file path to dump the labeled doc-sum pair
                default none. 

        in_memory: Bool, whether to return labeled samples in memory
                default False 

    return: list of triplets, [doc, sum, 0 or 1]
            0 if sum and doc do not match. 1 if so. 

    example: 
        >>> sample_generation.cross_pair([("A", "1"),("B", "2"), ("C", "3")], 1, in_memory=True)
            [('A', '1', 1),
             ('B', '2', 1),
             ('C', '3', 1),
             ('A', '2', 0),
             ('B', '1', 0),
             ('C', '1', 0)]

        >>> sample_generation.cross_pair([("A", "1"),("B", "2"), ("C", "3")], 20, in_memory=True)
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
    import random 

    samples = []

    # positive samples 
    if dump_to != None:
        f = open(dump_to, 'w')
        for (_doc, _sum) in data_pairs: 
            f.write("\t".join([_doc, _sum, "1\n"]))

    if in_memory:
        samples = [(_doc, _sum, 1) for (_doc,_sum) in data_pairs]

    # negative samples
    num_pair = len(data_pairs)

    for i in range(num_pair):
        doc_index = i
        legit_indexes = list(range(num_pair))
        legit_indexes.remove(i)
        sum_indexes = random.sample(legit_indexes, min(num_pair-1, neg_pos_ratio))
        for sum_index in sum_indexes:
            _doc, _sum = data_pairs[doc_index][0], data_pairs[sum_index][1]
            if dump_to != None:
                f.write("\t".join([_doc, _sum, "0\n"]))
            if in_memory:
                samples.append((_doc,_sum, 0))

    if dump_to != None:
        f.close()
    
    return samples

### mutation and associated functions 

def get_vocab(data_pairs):
    """generate a set of all vocabularies, tokenized by spaces, from pairs of documents and summaries  
    """
    long_doc, long_sum = "", "" 
    for (_doc, _sum) in data_pairs:
        long_doc += _doc + " "
        long_sum += _sum + " "
    all_sent = long_doc + long_sum 
    vocab = set(all_sent.split(' '))
    vocab.remove("")
    return vocab 

def mutate_add(words, vocab, ratio):
    """add _ratio_% of vocab into random locations in _words_

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, 
    ratio: float, 0 to 1. 

    example: 
        >>> mutate_add("i am a happy guy now".split(' '), ["hhhhh","jjjjj"], 0.2)
        'i am a happy jjjjj guy now'

    """
    import random
    length = len(words)
    indices = random.sample(range(length), int(ratio * length))
    res = []
    for i in range(length):
        if i in indices:
            res.append(vocab[random.randrange(len(vocab))])
        res.append(words[i])
    return ' '.join(res)

def mutate_delete(words, ratio, sent_end):
    """delete _ratio_% of words in random locations of _words_, 
    while preserving sentence separators

    words: list of strings, e.g., ["I", "am", "lucky"]
    ratio: float, 0 to 1 
    sent_end: list of strings

    example: 
        >>> mutate_delete("i am a happy guy now".split(' '), 0.2, ["."])
        'am a guy now'


    """
    import random
    length = len(words)
    indices = random.sample(range(length), int( (1 - ratio) * length))

    return ' '.join([words[i] for i in range(length) if i in indices and words[i] not in sent_end])

def mutate_replace(words, vocab, ratio, sent_end):
    """replace _ratio_% of words in random locations of _words_, 
    while preserving sentence separators

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, 
    ratio: float, 0 to 1  
    sent_end: list of strings


    example:
        >>> mutate_replace("i am a happy guy now".split(' '), ["hhhhh", "jjjjjjj"], 0.2, ["."]
    ...: )
        'i am a jjjjjjj guy now'

    """
    import random
    length = len(words)
    indices = random.sample(range(length), int(ratio * length))
    for i in indices: 
        if words[i] not in sent_end:
            words[i] =vocab[random.randrange(len(vocab))]
    return ' '.join(words)

def mutate(data_pairs, ratios, method, sent_end, dump_to=None, in_memory=False ):
    """Create positive and negative samples using cross pairing 

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary) 

        ratios: list of floats, each of which is 0 to 1. The ratio of mutation. 

        method: str, one of ["add", "delete", "replace"]

        sent_end: list of strings

        dump_to: str, file path to dump the labeled doc-sum pair
                default none. 

        in_memory: Bool, whether to return labeled samples in memory
                default False 

    return: list of triplets, [doc, mutated sum, ratio]
         
    example: 
         >>> mutate([("A B", "1 2"), ("C D", "3 4")], [0.5], 'replace', ['.'], in_memory= True )
            [('A B', '1 D', 0.5), ('C D', '1 4', 0.5)]

        >>> mutate([("A B", "1 2"), ("C D", "3 4")], [0.5], 'add', ['.'], in_memory= True )
            [('A B', '1 A 2', 0.5), ('C D', '3 A 4', 0.5)]

        >>> mutate([("A B", "1 2"), ("C D", "3 4")], [0.5], 'delete', ['.'], in_memory= True )
            [('A B', '1', 0.5), ('C D', '4', 0.5)]

    """
    import random 
    all_vocab = list(get_vocab(data_pairs))
    mutated = []

    if dump_to != None:
        f= open(dump_to, 'w') 

    for (_doc, _sum) in data_pairs:
        # split the words and then feed to mutator 
        splitted_summary = _sum.split(' ')

#        print ("generating samples using {} from dataset {}'s {} set"\
#              .format(method, "", split))

        for ratio in ratios:
            if method == "add":
                mutated_tmp = mutate_add(splitted_summary, all_vocab, ratio)
            elif  method == "delete":
                mutated_tmp = mutate_delete(splitted_summary, ratio, sent_end)
            elif  method == "replace":
                mutated_tmp = mutate_replace(splitted_summary, all_vocab, ratio, sent_end)
            else: 
                print ("wrong method of mutation")
                exit()

            if dump_to != None:
                f.write("\t".join([_doc, mutated_tmp, str(ratio)]))
                f.write("\n")
            if in_memory: 
                mutated.append((_doc, mutated_tmp, ratio))

    if dump_to != None:
        f.close()

    return mutated 

### put everything together 懶人包

def sample_generation():
    """

    2020-1-9: tested using pairs injected
    """

    import sample_conf as cfg
    import os
    dataset_name = cfg.dataset_name 
    for split in cfg.splits:
        pairs = load_pairs(cfg.dataset_name, split, cfg.take_percent, cfg.features, cfg.special_characters_to_clean)
#        pairs = [("A B", "1 2"), ("C D", "3 4")] # for testing
        for method in cfg.methods: 
            if method in ["add", "delete", "replace"]:
                # NOTE: vocabulary generation is repeated here
                samples = mutate(pairs, cfg.mutate_ratios, method, cfg.sent_end, dump_to=eval(cfg.dump_to), in_memory = cfg.in_memory)
            elif method in ["cross"]:
                print ("generating samples using {} from dataset {}'s {} set"\
                      .format(method, dataset_name, split))
                samples = cross_pair(pairs, cfg.neg_pos_ratio, dump_to=eval(cfg.dump_to), in_memory=cfg.in_memory)

    return samples # only the last one 


if __name__ == "__main__":
    sample_generation()


