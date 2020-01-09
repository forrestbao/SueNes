import tensorflow_datasets as tfds 

def replace_special_character(s, L):
    """replaces all special characters in _L_ within _s_ by spaces
    """
    for l in L:
        s= s.replace(l, " ")
    return s


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

def cross_pair(data_pairs, dump_to=None, in_memory=False ):
    """Create positive and negative samples using cross pairing 

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary)

        dump_to: str, file path to dump the labeled doc-sum pair
                default none. 

        in_memory: Bool, whether to return labeled samples in memory
                default False 

    return: list of triplets, [doc, sum, 0 or 1]
            0 if sum and doc do not match. 1 if so. 

    example: 
        >>> cross_pair([("1", "2"),("3", "4")], in_memory=True)
        [('1', '2', 1), ('3', '4', 1), ('1', '4', 0), ('3','2', 0)]

        >>> cross_pair([("1", "2"),("3", "4"), ("5", "6")], in_memory=True)
        [('1', '2', 1),
         ('1', '4', 0),
         ('1', '6', 0),
         ('3', '2', 0),
         ('3', '4', 1),
         ('3', '6', 0),
         ('5', '2', 0),
         ('5', '4', 0),
         ('5', '6', 1)]

    """

    samples = []

    if dump_to != None:
        f = open(dump_to, 'w')

    for doc_index in range(len(data_pairs)):
        _doc = data_pairs[doc_index][0]
        for sum_index in range(len(data_pairs)):
            _sum = data_pairs[sum_index][1] 
            if doc_index == sum_index: 
                label = 1 
            else: 
                label = 0 
            if dump_to != None:
                f.write("\t".join([_doc, _sum, str(label)+"\n"]))
            if in_memory :
                samples.append((_doc,_sum, label ))

    if dump_to != None:
        f.close()
    return samples

### functions related to mutation 

def get_vocab(data_pairs):
    """generate a set of all vocabularies, tokenized by spaces, from pairs of documents and summaries  
    """
    long_doc, long_sum = "", "" 
    for (_doc, _sum) in data_pairs:
        long_doc += _doc
        long_sum += _sum 
    all_sent = long_doc + long_sum 
    vocab = set(all_sent.split(' '))
    return vocab 

def mutate_add(words, vocab, ratio):
    """add _ratio_% of vocab into random locations in _words_

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, 
    ratio: int, 0 to 100. 

    """
    ratio /= 100. 
    length = len(words)
    indices = set([random.randint(0, length)
                   for _ in range(int((1 - ratio) * length))])
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
    ratio: int, 0 to 100. 
    sent_end: list of strings 
    """
    ratio /= 100.
    length = len(words)
    indices = set(random.sample(range(length),
                                int((1 - ratio) * length)))
    return ' '.join([words[i] for i in range(length)
                     if i not in indices or words[i] in sent_end])

def mutate_replace(words, vocab, ratio, sent_end):
    """replace _ratio_% of words in random locations of _words_, 
    while preserving sentence separators

    words: list of strings, e.g., ["I", "am", "lucky"]
    vocab: list of strings, 
    ratio: int, 0 to 100. 
    sent_end: list of strings 
    """

    ratio /= 100. 
    length = len(words)
    indices = set([random.randint(0, length)
                   for _ in range(int((1 - ratio) * length))])
    res = []
    for i in range(length):
        if i in indices and words[i] not in sent_end:
            res.append(vocab[random.randrange(len(vocab))] )
        else:
            res.append(words[i])
    return ' '.join(res)

def mutate(data_pairs, ratio, method,  dump_to=None, in_memory=False ):
    """Create positive and negative samples using cross pairing 

    input:
        data_pairs: list of 2-tuples of strings (a document, its summary) 

        ratio: int, 0 to 100. The percentage of mutation. 

        method: str, one of ["add", "delete", "replace"]

        dump_to: str, file path to dump the labeled doc-sum pair
                default none. 

        in_memory: Bool, whether to return labeled samples in memory
                default False 

    return: list of triplets, [doc, mutated sum, ratio]
         
    """
    import random 
    all_vocab = get_vocab(data_pairs)
    mutated = []

    if dump_to != None: 
        f= open(dump_to, 'w') 

    for (_doc, _sum) in data_pairs:
        # split the words and then feed to mutator 
        splitted_summary = _sum.split(' ')
        if method == "add":
            mutated_tmp = mutate_add(splitted_summary, all_vocab, ratio)
        elif  method == "delete":
            mutated_tmp = mutate_delete(splitted_summary, ratio, sent_end)
        elif  method == "replace":
            mutated_tmp = mutate_add(splitted_summary, all_vocab, ratio, sent_end)
        else: 
            print ("wrong method of mutation")
            exit()

        if dump_to != None:
            f.write("\t".join([_doc, _sum, str(ratio)]))
            
        if in_memory: 
            mutated.append(mutated_tmp)

    if dump_to != None:
        f.close()

    return mutated 

### functions at highest levels

def lazy_mutated(methods):
    import data_conf
    for split in data_conf.splits:
        pairs = load_pairs(data_conf.dataset_name, split, data_conf.take_percent, data_conf.features, data_conf.special_characters_to_clean)
        for method in data_conf.methods: 
            samples = mutate(pairs, data_conf.ratio, method, dump_to=eval(data_conf.dump_to), in_memory = data_conf.in_memory)


    return samples # only the last one 


def lan_ren_bao():
    import data_conf
    method = data_conf.cross_method[0] 

    for split in data_conf.splits:
        pairs = load_pairs(data_conf.dataset_name, split, data_conf.take_percent, data_conf.features, data_conf.special_characters_to_clean)
        samples = cross_pair(pairs, dump_to=eval(data_conf.dump_to), in_memory=data_conf.in_memory)

    return samples # only the last one

if __name__ == "__main__":
    samples = lan_ren_bao()


