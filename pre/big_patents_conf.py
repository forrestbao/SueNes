import string

# ==== constants about datasets 
dataset_features = {"cnn_dailymail": ['article', 'highlights'],
    "big_patent/all:1.0.0": ['description', 'abstract'],
    "newsroom": ['text', 'summary'],
    "scientific_papers": ['article', 'abstract'],
    "billsum":['text','summary'],
    "dryrun":[]
    }

dataset_sizes = {"billsum":23455, "scientific_papers":215913, "newsroom":1212740, 
        "cnn_dailymail":311971, "big_patent":1341362}

#======== data loading parameters 

load_from = "tfds" # "tfds" or "tsv"

# Whether save the data loaded in TFDS to TSV,
# for computers without TF2 or TFDS only. 
# If load_from is tsv, no effect. 
# TODO: get rid of this option as we no longer need it 
save_tsv = False

# should we scramble the data
# only effective when load_from is tfds 
scramble = True

# Must match their names in TFDS 
# dataset_name = "dryrun" 
dataset_name = "big_patent/all:1.0.0"

features = dataset_features[dataset_name]

splits = ['test', 'validation', 'train']
# splits = ['train']
# note that billsum has no validation set

# Percentage of data to load from orignal size 
load_percent =  100  # int from 0 to 100; 100 means all;0 means none. 

# Amoung loaded, usually after scambling, use only 1/num_shards of this dataset.
# Not to confuse with load_percent.
# For example, we may load all data and only use 1/10 of it, 
# vs. load only the first 10 percent and use all of that 10 percent. 
# Set as 1 for using all data loaded. 
num_shards = 10 # For big patent, this number is 1/10  


#========= data output/dumping parameters 

# filename nomenclature to save dumped data with labels 
# e.g., ./cnn_dailymail/cross/{train,validate,test}.tsv
# Set as "None" (a str, not Nonetype) if you do not wanna dump but use in memory 
dump_to="'../data/'+dataset_name.split('/')[0] + '/' + method + '/'+split+'.tsv'" 
# dump_to = "None"

# whether to save samples as variables in the memory 
# default: false 
in_memory=False

n_jobs = 15

# compact or plain 
# plain is 3-column, doc, summary, target
# but plain may contain repeated docs, 
# which will cause extra time in sentence embedding (not applicable for BERT) 
# compact: small. easy for inspecting dump. Format per line: 
# doc, sum1, label1, sum2, label2, sum3, label3, ...

dump_format = "compact"

#========= NLP parameters

special_characters_to_clean = ['\n', '\t'] # replace such strings in raw data 

sent_end = [".", "!", "?"]  # symbols that represent the end of a sentence 
sent_end = string.punctuation

#========= negative sampling parameters 

# ratio between negative and positive samples
# minimal: 1 
neg_pos_ratio = 5 

# methods used to generate negative samples 
methods = ["cross", "word_add", "word_delete", "word_replace"] 
# methods = ["word_add", "word_delete", "word_replace"] 
