dataset_name = "cnn_dailymail"
splits = ['train', 'validation', 'test']
special_characters_to_clean = ['\n', '\t', '\''] # replace such strings with spaces in raw data 
dump_to="'./'+dataset_name+'_'+method+'_'+split+'.tsv'" # _split_ is an element of _splits_ above 
take_percent =  1  # range from 0 to 100; 100 means all; 0 means none. 
features =['article', 'highlights']
in_memory=False
sent_end = [".","!","?"]  # symbols that represent the end of a sentence 


# parameters for cross pairing
use_cross = False  # True to enable sample generation using cross pairing; False to disable
cross_method = ["cross"] 

# parameters for mutate 
use_mutate = True # True to enable sample generation using mutation; False to disable

mutate_ratios = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 0 to 1 
mutate_method = ["add", "delete", "replace"] # comment this line to disable 
