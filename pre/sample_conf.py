#======== data loading parameters 

# Must match their names in TFDS 
dataset_name = "cnn_dailymail"
splits = ['train', 'validation', 'test']
features =['article', 'highlights']

# Percentage of data from the split to be used to generate samples
take_percent =  1  # range from 0 to 100; 100 means all;0 means none. 

#========= data output/dumping parameters 

# filename nomenclature to save dumped data with labels 
# e.g., ./cnn_dailymail_cross_test.tsv means test data, 
# generated using crosspairing on cnn_dailymail dataset
dump_to="'./'+dataset_name+'_'+method+'_'+split+'.tsv'" 

# whether to save samples as variables in the memory 
# default: false 
in_memory=False


#========= NLP parameters

special_characters_to_clean = ['\n', '\t', '\''] # replace such strings with spaces in raw data 

sent_end = [".","!","?"]  # symbols that represent the end of a sentence 

#========= negative sampling parameters 

# methods used to generate negative samples 
methods = ["cross", "add", "delete", "replace"] 

mutate_ratios = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 0 to 1 
