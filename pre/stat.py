import random
import itertools, re, os
import copy
import functools
import random
import matplotlib.pyplot as plt
import seaborn as sns
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
            plt.figure()
            plt.hist(ns, bins=numpy.arange(15))
            plt.show()
            plt.close()
            print("Mean %.2f, Middle %d" % (numpy.mean(ns), numpy.median(ns)) )
            

def train_distribution(conf):
    
    cfg = __import__(conf)
    import sys
    sys.path.append('../bert')
    import utils
    
    for dataset_name in cfg.dataset_names:
        train_path = os.path.join('../data_old/', dataset_name, 'sent_delete')
        train_file = os.path.join(train_path, "train.tsv")
        # os.system("cat " + os.path.join(train_path, 'train_*.tsv') + ' > ' + train_file)

        lines = utils.DataProcessor._read_tsv(train_file)

        labels = []
        for line in lines: # line is already tab-separated 
            for j in range(1, len(line), 2) :
                label = float(line[j+1].strip())
                labels.append(label)
        
        print(dataset_name, len(labels))
        sns.kdeplot(labels, label=dataset_name)
    plt.legend()
    plt.show()
        


# sentence_splitter = init() # a global variable

if __name__ == "__main__":
    #sample_stat("sentence_conf")
    train_distribution("sentence_conf")
