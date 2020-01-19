#======== data loading parameters 

load_from = "tsv" # "tfds" or "tsv"

# Whether save the data loaded in TFDS to TSV. 
# If load_from is tsv, no effect  
save_tsv = True

# Must match their names in TFDS 
dataset_name = "cnn_dailymail"
dataset_features = {"cnn_dailymail": ['article', 'highlights'],
    "big_patent": ['description', 'abstract'],
    "newsroom": ['text', 'summary'],
    "scientific_papers": ['artcile', 'abstract'],
    "billsum":['text','summary']
    }

dataset_sizes = {"billsum":23455, "scientific_papers":215913, "newsroom":1212740, 
        "cnn_dailymail":311971, "big_patent":1341362}

splits = ['train', 'validation', 'test']
features = dataset_features[dataset_name]

# Percentage of data from the split to be used to generate samples
take_percent =  1  # range from 0 to 100; 100 means all;0 means none. 

#========= data output/dumping parameters 

# filename nomenclature to save dumped data with labels 
# e.g., ./cnn_dailymail_cross_test.tsv means test data, 
# generated using crosspairing on cnn_dailymail dataset
#dump_to="'./'+dataset_name+'_'+method+'_'+split+'_'+take_percent+'.tsv'" 
dump_to="'./'+dataset_name+'_'+method+'_'+split+'.tsv'" 

# whether to save samples as variables in the memory 
# default: false 
in_memory=False


#========= NLP parameters

special_characters_to_clean = ['\n', '\t', '\''] # replace such strings with spaces in raw data 

sent_end = [".","!","?"]  # symbols that represent the end of a sentence 

#========= negative sampling parameters 

# ratio between negative and positive samples; for cross-pairing only 
neg_pos_ratio = 5 

# methods used to generate negative samples 
methods = ["cross","add", "delete", "replace"] 
#methods = ["delete"]

mutate_ratios = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 0 to 1 
