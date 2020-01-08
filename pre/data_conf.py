dataset_name = "cnn_dailymail"
splits = ['train', 'validation', 'test']
special_characters_to_clean = ['\n', '\t', '\''] # replace such strings with spaces in raw data 
dump_to="'./'+split+'.tsv'" # _split_ is an element of _splits_ above 
take_percent =  1  # range from 0 to 100; 100 means all; 0 means none. 
features =['article', 'highlights']
in_memory=False

