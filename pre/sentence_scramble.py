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

    
# def replace(i, num_samples, lengths_summaries):
#     """generate indexes for replacing sentences in a summary
#     """
#     ratios = [random.uniform(0, 1) for _ in range(neg_pos_ratio)]
#     length = lengths_summaries[i] # number of sentences in i-th summary

#     for ratio in ratios:
#         indices = random.sample(range(length), int(ratio * length)) # summary sentences to replace 
#         for index in indices:
#             random_summary_sentence = random.randint(0, num_samples)




# def replace(pairs, neg_pos_ratio):
#     """Replace sentences in a summary
#     """

#     num_samples = len(pairs)
#     lengths_summaries = list(map(len, zip(*pairs)[1]))# number of sentences in each summary


    

#     ratios = [random.uniform(0, 1) for _ in range(neg_pos_ratio)]

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
    >>> x = delete([("doc1", range(10)), ('doc2', range(10,20))],  3)
    >>> x
    [['doc1', range(0, 10), 0, 
              [0, 3, 4, 5, 7, 8], 0.40414900308686985, 
              [0, 3, 4, 5, 6, 8], 0.44639878562443647, 
              [6, 8, 9], 0.018728424512874597], 
     ['doc2', range(10, 20), 0, 
              [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 0.9393603341915923, 
              [12, 13, 15, 17, 19], 0.4337312018927616, 
              [10, 12, 13, 14, 15, 16, 18], 0.5925348876612918]
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
        line = [_doc, _sum, 0] # document, positive sample, and ratio 0 (nothing deleted)

        indexes_of_sentences_to_delete = \
        [numpy.random.randint(0, lengths_summaries[sample_id], size=i) for i in delete_numbers[sample_id]] 

        for i in range(neg_pos_ratio): 
            # number_of_sentences_to_delete_in_this_neg_sample  = 
            indexes_of_sentences_to_delete = numpy.random.randint(0, lengths_summaries[sample_id], size= delete_numbers[sample_id][i])

            _sum_alternative = [_sum[i] for i in range(len(_sum)) if i not in indexes_of_sentences_to_delete]

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
            # lines = delete(pairs[start:end], neg_pos_ratio)
            lines = delete(pairs, neg_pos_ratio)

        if debug:
            print (lines)

        print ("Dumping into", dumpfile, end="...")
        with open(dumpfile, 'w') as f:
            for line in lines:
                line = map(str, line)
                f.write("\t".join(line))

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