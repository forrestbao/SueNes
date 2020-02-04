# ==== constants about datasets 
dataset_features = {"cnn_dailymail": ['article', 'highlights'],
    "big_patent": ['description', 'abstract'],
    "newsroom": ['text', 'summary'],
    "scientific_papers": ['article', 'abstract'],
    "billsum":['text','summary']
    }

dataset_sizes = {"billsum":23455, "scientific_papers":215913, "newsroom":1212740, 
        "cnn_dailymail":311971, "big_patent":1341362}

#======== data loading parameters 

load_from = "tfds" # "tfds" or "tsv"

# Whether save the data loaded in TFDS to TSV,
# for computers without TF2 or TFDS only. 
# If load_from is tsv, no effect. 
save_tsv = False

# should we scramble the data
# only effective when load_from is tfds 
# No need to scramble if load_percent is 100 and num_shards is 1
scramble = False 

# Must match their names in TFDS 
dataset_name = "cnn_dailymail"

features = dataset_features[dataset_name]

splits = ['train', 'test', 'validation']
# note that billsum has no validation set

# Percentage of data to load from orignal size 
load_percent =  100  # range from 0 to 100; 100 means all;0 means none. 

# Amoung loaded, usually after scambling, use only 1/num_shards of this dataset.
# Not to confuse with take_percent.
# for example, we may load all data and only use 1/10 of it, 
# vs. load only the first 10 percent and use all of that 10 percent. 
# Set as 1 for using all data loaded
num_shards = 1 


#========= data output/dumping parameters 

# filename nomenclature to save dumped data with labels 
# e.g., ./cnn_dailymail_cross_test.tsv means test data, 
# generated using crosspairing on cnn_dailymail dataset
# Set as None if you do not wanna dump but use in memory 
dump_to="'./'+dataset_name+'_'+method+'_'+split+'.tsv'" 

# whether to save samples as variables in the memory 
# default: false 
in_memory=False

n_jobs = 16


#========= NLP parameters

special_characters_to_clean = ['\n', '\t', '\''] # replace such strings in raw data 

sent_end = [".","!","?"]  # symbols that represent the end of a sentence 

#========= negative sampling parameters 

# ratio between negative and positive samples; for cross-pairing only 
neg_pos_ratio = 5 

# methods used to generate negative samples 
methods = ["cross","add", "delete", "replace"] 
# methods = ["cross"]
# methods = ["add", "delete", "replace"] 

mutate_ratios = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 0 to 1 
