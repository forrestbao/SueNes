import random
import itertools, re, os
import copy
import functools
import random
from tqdm import tqdm

import numpy
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds

import time

from sample_generation import auto_escape, replace_special_character, normalize_sentence 

def sample_stat(conf):
    """main function to generate samples 
    """

    cfg = __import__(conf)
    from nltk.tokenize import sent_tokenize
        

    for dataset_name in cfg.dataset_names:
        print ("From dataset:", dataset_name)
        features = cfg.dataset_features[dataset_name]

        for split in cfg.splits:
            print ("Data split:", split)
            total_samples = cfg.dataset_sizes_w_split[dataset_name][split]
            dataset = tfds.load(name=dataset_name, download=False, split=split)

            ns = []
            with tqdm(total=total_samples) as pbar:
                for sample in dataset:
                    # _doc = normalize_sentence(sample[features[0]].numpy().decode("utf-8"), cfg.special_characters_to_clean)
                    _sum = normalize_sentence(sample[features[1]].numpy().decode("utf-8"), cfg.special_characters_to_clean)
                    _sum_sents = sent_tokenize(_sum)
                    ns.append(len(_sum_sents))

                    pbar.update(1)

            print("Mean %.2f, Middle %d" % (numpy.mean(ns), numpy.median(ns)) )
            

# sentence_splitter = init() # a global variable

if __name__ == "__main__":
    sample_stat("sentence_conf")
