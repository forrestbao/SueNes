# Sentence-level negative sampling

import random
import itertools, re, os
import joblib, multiprocessing
import copy
import functools
import random

import numpy
import stanza 
import tensorflow_datasets as tfds

import time

from sample_generation import auto_escape, replace_special_character, normalize_sentence


def init(tokenize_batch_size = 64):
    """NOT IN USE. KEPT FOR REFERENCE
    """
    try: 
        sentence_splitter = stanza.Pipeline(lang='en', processors={'tokenize':"spacy"}, tokenize_batch_size=tokenize_batch_size)
    except Exception:
        stanza.download('en')
        sentence_splitter = stanza.Pipeline(lang='en', processors={'tokenize':"spacy"}, tokenize_batch_size=tokenize_batch_size)
    return sentence_splitter

# def split_sentence(paragraph, sentence_splitter)

def split_pairs(pairs, my_batch_size=2**10, tokenize_batch_size=64, n_jobs= 4):
    """For each pair, return the summary as a list of strings 

    Process-level parallelism impossible with Stanza
    # AttributeError: Can't pickle local object 'CNNClassifier.__init__.<locals>.<lambda>'

    Examples
    -------------
    >>> split_pairs([("iam ahjppy asf.","fsd. sdf.fsd. fsd. f")], 64)                                       
    [('iam ahjppy asf.', ['fsd.', 'sdf.fsd.', 'fsd.', 'f'])]
    >>> list(split_pairs([("i am ahjppy.", "today is monday. food is yummy."), ("the code is hard. the water is
     ...:  cold.", "the number is low. the government sucks. ")], 64))
    [('i am ahjppy.', ['today is monday.', 'food is yummy.']),
     ('the code is hard. the water is cold.',
      ['the number is low.', 'the government sucks.'])]
    """

    print ("Splitting summaries...", end= " ")

    try: 
        stanza_sentence_splitter = stanza.Pipeline(lang='en', processors='tokenize', tokenize_batch_size=tokenize_batch_size, verbose=False)
    except Exception:
        stanza.download('en')
        stanza_sentence_splitter = stanza.Pipeline(lang='en', processors='tokenize', tokenize_batch_size=tokenize_batch_size, verbose=False)


    # Stanza tokenization with delimiters.    
    list_summaries = []
    combined_summary = " Forrest loves Ames. ".join([_sum  for (_doc, _sum) in pairs])
    # This cannot be too long. otherwise, efficienty is very low. 

    summary_splitted = [x.text for x in stanza_sentence_splitter(combined_summary).sentences]

    # print (combined_summary)
    # print (summary_splitted)
    
    summary_temp = []
    for sent in summary_splitted:
        if len(sent) < 2:
            continue 
        elif sent != "Forrest loves Ames.":
            summary_temp.append(sent)
        else:
            # sent = sent[:-3]
            # summary_temp.append(sent)
            list_summaries.append(summary_temp)
            summary_temp = []
    list_summaries.append(summary_temp) # last one 

    new_pairs = list(zip([_doc for (_doc, _sum) in pairs], list_summaries))
    return new_pairs


def replace(pairs, neg_pos_ratio):
    """Replace sentences in a summary

    The code does something similar to delete() but using a slightly different strategy to vectorize 

    Examples
    -------------
    >>> x = replace([("doc1", [str(i) for i in  range(10) ] ), ('doc2', [str(i) for i in  range(10,20)])],  3) 
    [['doc1',
      '0 1 2 3 4 5 6 7 8 9',   1,
      '14 15 12 17 4 11 6 11 16 14',  0.2,
      '0 14 2 3 4 5 6 7 8 10',  0.8,
      '0 13 13 10 14 15 15 14 11 17',  0.1
     ],
     ['doc2',
      '10 11 12 13 14 15 16 17 18 19',  1,
      '9 9 4 13 0 15 16 3 0 3',   0.3,
      '10 2 1 13 9 15 7 3 18 19',   0.5,
      '0 1 9 13 14 6 9 5 18 2',  0.3
     ]
    ]



    """

    num_samples = len(pairs)
    lengths_summaries = list(map(len, list(zip(*pairs))[1]))# number of sentences in each summary

    # Prepare a flattened list of (doc-sum pair, summary_id) to randomly pick a summary to replace 
    # summary_id_flattened = [(pair_id, sentence_id) for pair_id, (_, summaries) in enumerate(pairs) for sentence_id in range(len(summaries)) ]
    # random.shuffle(summary_id_flattened)
    # summary_id_flattened = numpy.array(summary_id_flattened)

    # total_summaries = len(summary_id_flattened)

    # generate indexes of sentences to keep
    keep_ratios = numpy.random.rand(num_samples, neg_pos_ratio)
    # 2-D array, each row is the 0<ratio<1 of summary sentences to replace

    keep_numbers = numpy.einsum("ij, i->ij", keep_ratios, lengths_summaries)
    keep_numbers = keep_numbers.astype(int) 
    # how many sentences to keep in each neg-sampled summary, integers

    new_pairs = [] # compact form [_doc, _sum_1, label_1, _sum_2, _label_2, ....]
    for sample_id, (_doc, _sum) in enumerate(pairs):
        new_sums = [_doc, " ".join(_sum), 1]
        for i in range(neg_pos_ratio): # generate negative samples 
            keep_number = keep_numbers[sample_id][i]

            indexes_of_sentences_to_keep = random.sample(range(len(_sum)), k = keep_number)

            #----------wrong approach, joint probabilities 
            # label = numpy.random.rand()
            # replace_list = numpy.random.rand(len(_sum)) > label# Boolean 
            #---------------

            new_sum = copy.deepcopy(_sum)
            for sentence_id in range(len(_sum)):
                if sentence_id not in indexes_of_sentences_to_keep:
                    # --- won't work, may select sentences in this summary
                    # (pair_id, sentence_id_to_replace) = summary_id_flattened[random.randint(0, total_summaries-2)]
                    # new_sum[sentence_id] = pairs[pair_id][1][sentence_id_to_replace]

                    pair_id = sample_id # pick to new doc-sum pair
                    while pair_id == sample_id:
                        pair_id = random.randint(0, len(pairs)-1)
                    new_sum [sentence_id] = random.choice(pairs[pair_id][1])

            label = keep_number/len(_sum)
            # label = keep_ratios[sample_id][i] # NOTE alternative, introducing noise
            
            new_sum = " ".join(new_sum)
            new_sums += [new_sum, label]

        new_pairs.append(new_sums)
    return new_pairs

def delete(pairs, neg_pos_ratio):
    """Randomly delete sentences from sample 

    neg_pos_ratio: how many negative samples to generate 


    How the code works by taking as much matrix operation as possible
    ---------------------------------------------------------------------

    >>> delete_ratios =numpy.array([[0.1, 0.4, 0.8], [0.2, 0.5, 0.7]])
    >>> lengths_summaries=[4, 10]                
    >>> delete_numbers = numpy.einsum("ij, i->ij", delete_ratios, lengths_summaries)
    >>> delete_numbers # number of sentences to delete from summaries
    array([[0.4, 1.6, 3.2],
        [2. , 5. , 7. ]])
    >>> delete_numbers = delete_numbers.astype(int)
    >>> delete_numbers
    array([[0, 1, 3],
            [2, 5, 7]])
    >>> delete_indexes = [[numpy.random.randint(0, lengths_summaries[sample_id], size=i) for i in delete_numbers[sample_id]]  for sample_id in range(2)]
    >>> delete_indexes 
    [[array([]),
      array([2]),
      array([1, 1, 3])],
    [array([8, 5]),
     array([1, 6, 6, 9, 3]),
     array([0, 7, 8, 6, 2, 2, 8])]]
    

    https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy


    Examples 
    ------------
    >>> x = delete([("doc1", [str(i) for i in  range(10) ] ), ('doc2', [str(i) for i in  range(10,20)])],  3) 
    >>> x
    [['doc1',
      '0 1 2 3 4 5 6 7 8 9',      1,
      '0 1 3 7 8 9',        0.4146246255365821,
      '0 2 3 4 5 7',        0.49660148081187294,
      '0 1 2 3 4 5 8 9',        0.795436928305486
     ],
     ['doc2',
        '10 11 12 13 14 15 16 17 18 19',        1,
        '11 16 17 18',        0.2789769714907715,
        '11 12 13 16',         0.2966426208197248,
        '10 11 12 13 14 16 17 18 19',        0.8916115194294404
     ]
    ]


    In the example above, we just use numbers as sentences. 

    """

    num_samples = len(pairs)
    lengths_summaries = list(map(len, list(zip(*pairs))[1]))# number of sentences in each summary

    # generate index of sentences to delete 
    delete_ratios = numpy.random.rand(num_samples, neg_pos_ratio)
    # 2-D array, each row is the 0<ratio<1 of summary sentences to delete

    delete_numbers = numpy.einsum("ij, i->ij", delete_ratios, lengths_summaries)
    delete_numbers = delete_numbers.astype(int) 
    # how many sentences to delete in each summary, integers
        
    # delete_indexes = [[numpy.random.randint(0, lengths_summaries[sample_id], size=i, dtype='int32') for i in delete_number[sample_id]]  for sample_id in range(2)]

    # combine into the output format
    lines = []
    for sample_id, (_doc, _sum) in enumerate(pairs):
        line = [_doc, " ".join(_sum), 1.] # document, positive sample, and ratio 0 (nothing deleted)

        indexes_of_sentences_to_delete = \
        [numpy.random.randint(0, len(_sum), size=i) for i in delete_numbers[sample_id]] 

        for i in range(neg_pos_ratio): 
            # number_of_sentences_to_delete_in_this_neg_sample  = 
            indexes_of_sentences_to_delete = numpy.random.randint(0, lengths_summaries[sample_id], size= delete_numbers[sample_id][i])

            _sum_alternative = [_sum[i] for i in range(len(_sum)) if i not in indexes_of_sentences_to_delete]
            _sum_alternative = " ".join(_sum_alternative)

            _ratio = delete_ratios[sample_id][i]
            line += [_sum_alternative, 1 - _ratio]

        lines.append(line)
    return lines  

def mutate(pairs, method, dumpfile, neg_pos_ratio, batch_size= 2**12, debug=False):
        """Central method for delete and replace, bacthed 
        """

        print ("Mutating", end="...")

    # boundaries = list(range(0, len(pairs), batch_size))
    # boundaries.append(len(pairs))
    # boundaries = list(zip(boundaries[:-1], boundaries[1:]))

    # for i, (start, end) in enumerate(boundaries):
    #     print ("mutating, batch {0}/{1}".format(i+1, len(boundaries)))

        if method == 'delete':
            lines = delete(pairs, neg_pos_ratio)
        elif method == 'replace':
            lines = replace(pairs, neg_pos_ratio)

        if debug:
            print (lines)

        print ("Dumping into", dumpfile, end="...")
        with open(dumpfile, 'w') as f:
            for line in lines:
                line = map(str, line)
                f.write("\t".join(line)+"\n")

def generate_one(dataset_name, split, features, methods, neg_pos_ratio, load_start, load_end, special_chars, data_root, batch_id): 
    """Generate one batch of data for one split (test or train) on one dataset, 
    given the start and end indexes of samples in the dataset
    """

    # 1. Load data 
    dataset = tfds.load(name=dataset_name, download=False, 
                        split=split+ '[{}:{}]'.format(load_start, load_end)
                       )

    pairs = [(normalize_sentence(piece[features[0]].numpy().decode("utf-8"), special_chars), 
              normalize_sentence(piece[features[1]].numpy().decode("utf-8"), special_chars) )
                for piece in dataset]

    # 2. Split summary sentences 
    pairs = split_pairs(pairs)

    for method in methods:
        # 3. Mutate and write results to file 
        dumpfile = os.path.join(data_root, dataset_name, method, "{}_{}.tsv".format(split, batch_id))
        if not os.path.exists(os.path.dirname(dumpfile)):
            try:
                os.makedirs(os.path.dirname(dumpfile))
            except OSError as exc: # Guard against rare conditions
                if exc.errno != errno.EEXIST:
                    raise

        mutate(pairs, method, dumpfile, neg_pos_ratio, batch_size=2**11)   

def sample_generation(conf):
    """main function to generate samples 
    """

    cfg = __import__(conf)

    for dataset_name in cfg.dataset_names:
        print ("From dataset:", dataset_name)
        features = cfg.dataset_features[dataset_name]

        for split in cfg.splits:
            print ("Data split:", split)
            total_samples = cfg.dataset_sizes_w_split[dataset_name][split]

            boundaries = list(range(0, total_samples, cfg.my_batch_size))
            boundaries.append(total_samples)
            boundaries = list(zip(boundaries[:-1], boundaries[1:]))

            for batch_id, (load_start, load_end) in enumerate(boundaries):
                print ("\t batch {0}/{1}".format(batch_id+1, len(boundaries)), end="...")
                start_time = time.time()
                generate_one(dataset_name, split, features, cfg.methods, cfg.neg_pos_ratio, load_start, load_end, cfg.special_characters_to_clean, cfg.data_root, batch_id)

                elapse = time.time() - start_time
                print ("  Took {:.3f} seconds".format(elapse))


# sentence_splitter = init() # a global variable

if __name__ == "__main__":
    sample_generation("sentence_conf")