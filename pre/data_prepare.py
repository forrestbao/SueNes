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
        data_pairs: list of 2-tuples (a document, its summary)

        dump_to: str, file path to dump the labeled doc-sum pair
                default none. 

        in_memory: Bool, whether to return labeled samples in memory
                default False 

    return: list of triplets, [doc, sum, 0 or 1]
            0 if sum and doc do not match. 1 if so. 

    """

    pos_samples, neg_samples = None, None 

    if dump_to != None:
        with open(dump_to, 'w') as f: 
            for (_doc, _sum) in data_pairs:
                f.write("\t".join([_doc, _sum, "1\n"]))

            for doc_index in range(len(data_pairs)):
                _doc = data_pairs[doc_index][0]
                for sum_index in range(doc_index+1, len(data_pairs)):
                    _sum = data_pairs[sum_index][1]
                f.write("\t".join([_doc, _sum, "0\n"]))

        
    if in_memory: 
        pos_samples = [(_doc, _sum, 1) for (_doc, _sum) in data_pairs ]

        neg_samples = [] 
        for doc_index in range(len(data_pairs)):
            _doc = data_pairs[doc_index][0]
            for sum_index in range(doc_index+1, len(data_pairs)):
                _sum = data_pairs[sum_index][1]
                neg_samples.append((_doc,_sum, 0 ))

    return pos_samples, neg_samples 

def mutate(data_pairs):
    pass


def lan_ren_bao():
    import data_conf

    for split in data_conf.splits:
        pairs = load_pairs(data_conf.dataset_name, split, data_conf.take_percent, data_conf.features, data_conf.special_characters_to_clean)
        pos_samples, neg_samples = cross_pair(pairs, dump_to=eval(data_conf.dump_to), in_memory=data_conf.in_memory)

    return pairs

if __name__ == "__main__":
    pairs = lan_ren_bao()


