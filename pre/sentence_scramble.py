# Sentence-level negative sampling

import random
import itertools, re, os
import joblib, multiprocessing
import copy
import functools
import random
import errno 

import numpy
import stanza
import spacy
import tensorflow_datasets as tfds

import time

from sample_generation import auto_escape, replace_special_character, normalize_sentence

def split_pairs(pairs, tokenizer_name="spacy", spacy_batch_size=2**10, n_jobs= 4):
    """For each pair, return the summary as a list of strings 

    tokenizer_name: str, "spacy", "stanza", or "nltk"
    Spacy is about 4 times faster than Stanza --- not fully saturated CPU

    Process-level parallelism impossible with Stanza
    # AttributeError: Can't pickle local object 'CNNClassifier.__init__.<locals>.<lambda>'

    FIXME: interestingly, spacy_batch_size and n_jobs have little effect on Spacy's speed

    Examples
    -------------
    >>> split_pairs([("iam ahjppy asf.","fsd. sdf.fsd. fsd. f")])
    [('iam ahjppy asf.', ['fsd.', 'sdf.fsd.', 'fsd.', 'f'])]
    >>> split_pairs([("i am ahjppy.", "today is monday. food is yummy."), ("the code is hard. the water is cold.", "the number is low. the government sucks. ")], )
    [('i am ahjppy.', ['today is monday.', 'food is yummy.']),
     ('the code is hard. the water is cold.',
      ['the number is low.', 'the government sucks.'])]
    """

    print ("Splitting summaries...", end= " ")

    if tokenizer_name == "stanza":
        # Tokenization with delimiters.    
        list_summaries = []
        combined_summary = " Forrest loves Ames. ".join([_sum  for (_doc, _sum) in pairs])

        try: 
            nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)
        except Exception:
            stanza.download('en')
            nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)

        # This cannot be too long. otherwise, efficienty is very low. 

        summary_splitted = [x.text for x in nlp(combined_summary).sentences]

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

    elif tokenizer_name == 'spacy':
            nlp=spacy.load("en_core_web_sm", exclude=["tok2vec",'tagger','parser','ner', 'attribute_ruler', 'lemmatizer'])
            nlp.add_pipe("sentencizer")
            nlp.max_length = 2000000 # default is 1,000,000

            list_summaries = [
                [x.text for x in doc.sents] # sentences in each summary
                for doc in nlp.pipe( list(zip(*pairs)) [1], n_process= n_jobs, batch_size=spacy_batch_size)]
    
    elif tokenizer_name == 'nltk':
        from nltk.tokenize import sent_tokenize
        list_summaries = [sent_tokenize(_sum) for (_doc, _sum) in pairs]

    new_pairs = list(zip(
                    list(zip(*pairs))[0], # docs 
                    list_summaries # list of list of str, segmented summaries
                ))
    return new_pairs

def mutate(pairs, method, dumpfile, neg_pos_ratio, mode='len',debug=False):
    """Central method for delete and replace

    The mutation step is much faster than sentence segmentation.
    E.g., 0.005 second vs 3 seconds. 
    So it's not parallelized. 

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
    -------------
    >>> sentence_scramble.mutate([("doc1", [str(i) for i in  range(10) ] ), ('doc2', [str(i) for i in  range(10,20)])],  "sent_delete", "/dev/null", 3)
    [['doc1',
      '0 1 2 3 4 5 6 7 8 9',    1.0,
      '0 2 3 4 6 7 8 9',    0.8,
      '9',     0.1,
      '0 1 2 3 4 5 6 7 8',    0.9],
     ['doc2',
      '10 11 12 13 14 15 16 17 18 19',    1.0,
      '13 14 15',    0.3,
      '10 11 14',    0.3,
      '10 11 13 14 16 17 18 19',   0.8
     ]
    ]

    >>> sentence_scramble.mutate([("doc1", [str(i) for i in  range(10) ] ), ('doc2', [str(i) for i in  range(10,20)])],  "sent_replace", "/dev/null", 3)
    [['doc1',
      '0 1 2 3 4 5 6 7 8 9',    1.0,
      '19 19 2 10 4 5 13 12 18 18',    0.3,
      '13 14 16 14 11 11 18 18 12 15',    0.0,
      '12 1 14 11 13 15 16 18 11 18',    0.1],
     ['doc2',
      '10 11 12 13 14 15 16 17 18 19',    1.0,
      '7 0 5 4 5 0 9 6 4 1',    0.0,
      '6 2 1 9 7 4 2 8 1 0',    0.0,
      '8 4 3 13 14 15 16 2 9 7',    0.4
     ]
    ]

    """

    print ("Mutating", end="...")

    num_samples = len(pairs)
    lengths_summaries = list(map(len, list(zip(*pairs))[1]))# number of sentences in each summary
    
    # generate indexes of sentences to keep
    keep_ratios = numpy.random.rand(num_samples, neg_pos_ratio)
    # 2-D array, each row is the 0<ratio<1 of summary sentences to keep

    keep_numbers = numpy.einsum("ij, i->ij", keep_ratios, lengths_summaries)
    keep_numbers = keep_numbers.astype(int)  # round to integers
    # TODO: can we inject noise here? 
    # how many sentences to keep in each neg-sampled summary, integers

    lines = [] # compact form [_doc, _sum_1, label_1, _sum_2, _label_2, ....]
    for sample_id, (_doc, _sum) in enumerate(pairs):
        line = [_doc, " ".join(_sum), 1.0]
        full_summary_length = len(" ".join(_sum))
        for i in range(neg_pos_ratio): # generate negative samples 
            keep_number = keep_numbers[sample_id][i]
            if keep_number < 1: # at least 1 sentence needed
                continue

            indexes_of_sentences_to_keep = random.sample(range(len(_sum)), k = keep_number)

            if method == "sent_replace":
                new_sum = copy.deepcopy(_sum)
                for sentence_id in range(len(_sum)):
                    if sentence_id not in indexes_of_sentences_to_keep:

                        pair_id = sample_id # pick to new doc-sum pair
                        while pair_id == sample_id:
                            pair_id = random.randint(0, len(pairs)-1)
                        new_sum [sentence_id] = random.choice(pairs[pair_id][1])
            elif method == "sent_delete":
                new_sum = [_sum[i] for i in range(len(_sum)) if i in indexes_of_sentences_to_keep]

            
            new_sum = " ".join(new_sum)
            label = 0
            if mode == 'sent':
                label = keep_number/len(_sum)
            elif mode == 'char': 
                label = len(new_sum) / float(full_summary_length) 
            # label = keep_ratios[sample_id][i] # NOTE alternative, introducing noise            
            line += [new_sum, label]

        lines.append(line)

    if debug:
        print (lines)

    print ("Dumping into", dumpfile, end="...")
    with open(dumpfile, 'w') as f:
        for line in lines:
            line = map(str, line)
            f.write("\t".join(line)+"\n")

    return lines

def generate_one(dataset_name, split, features, methods, neg_pos_ratio, load_start, load_end, special_chars, data_root, tokenizer_name, n_jobs, spacy_batch_size, batch_id, mode): 
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

    pairs = [(_doc, _sum) for (_doc, _sum) in pairs if len(_sum) > 0]

    # 2. Split summary sentences 
    pairs = split_pairs(pairs, tokenizer_name=tokenizer_name, spacy_batch_size=spacy_batch_size, n_jobs=n_jobs)

    for method in methods:
        # 3. Mutate and write results to file 
        dumpfile = os.path.join(data_root, dataset_name, method + "_" + mode, "{}_{}.tsv".format(split, batch_id))
        if not os.path.exists(os.path.dirname(dumpfile)):
            try:
                os.makedirs(os.path.dirname(dumpfile))
            except OSError as exc: # Guard against rare conditions
                if exc.errno != errno.EEXIST:
                    raise

        mutate(pairs, method, dumpfile, neg_pos_ratio, mode)

def combine_shuffle(methods, data_root, dataset_name, split, mode):
    """Combine dumped sample files into one file and shuffle

    cat train_*.tsv > train.tsv
    rm  train_*.tsv 
    cat test_*.tsv > test.tsv
    rm test_*.tsv 
    shuf train.tsv  > train_shuffled.tsv
    shuf test.tsv  > test_shuffled.tsv
    head train_shuffled.tsv -n 120722 > train_shuffled_10_percent.tsv
    head test_shuffled.tsv -n 6707 > test_shuffled_10_percent.tsv


    """
    for method in methods:
        dump_root = os.path.join(data_root, dataset_name, method + "_" +  mode)
        print ("Combining and shuffling at ", dump_root)
        chops = os.path.join(dump_root, F"{split}_*.tsv")
        concrete = os.path.join(dump_root, F"{split}.tsv")
        tmp = os.path.join(dump_root, "tmp.tsv")

        os.system(F"cat {chops} > {concrete}")
        os.system(F"rm {chops}")
        os.system(F"shuf {concrete} > {tmp}")
        os.system(F"mv {tmp} {concrete}")

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
                print ("\t batch {0}/{1}".format(batch_id+1, len(boundaries)), end="...", flush=True)
                start_time = time.time()
                generate_one(dataset_name, split, features, cfg.methods, cfg.neg_pos_ratio, load_start, load_end, cfg.special_characters_to_clean, cfg.data_root, cfg.tokenizer_name, cfg.n_jobs, cfg.spacy_batch_size, batch_id, cfg.mode)

                elapse = time.time() - start_time
                print ("  Took {:.3f} seconds".format(elapse))

            combine_shuffle(cfg.methods, cfg.data_root, dataset_name, split, cfg.mode)


if __name__ == "__main__":
    sample_generation("sentence_conf")