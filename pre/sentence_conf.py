import string

# ==== constants about datasets 
dataset_features = {"cnn_dailymail": ['article', 'highlights'],
    "big_patent": ['description', 'abstract'],
    "newsroom": ['text', 'summary'],
    "scientific_papers": ['article', 'abstract'],
    "billsum":['text','summary'],
    "dryrun":[]
    }

dataset_sizes = {"billsum":23455, "scientific_papers":215913, "newsroom":1212740, 
                 "cnn_dailymail":311971, "big_patent":1341362}

dataset_sizes_w_split = {# new for sentence-level mutation
    "billsum":{'train':18949, 'test':3269},   
    "cnn_dailymail":{'train':287113, 'test':11490},
    "big_patent":{'train':1207222, 'test':67702},
    "scientific_papers":{'train':203037, 'test':6440},
}

#======== data loading parameters 

# Must match their names in TFDS 
# dataset_name = "dryrun" 
dataset_names = ["billsum", "cnn_dailymail", "scientific_papers", "big_patent"]

splits = ['train', 'test']
# note that billsum has no validation set

#========= data output/dumping parameters 

data_root = "/mnt/12T/data/NLP/anti-rogue/data/"  # new for sentence-level mutation


n_jobs = 35

# compact or plain 
# plain is 3-column, doc, summary, target
# but plain may contain repeated docs, 
# which will cause extra time in sentence embedding (not applicable for BERT) 
# compact: small. easy for inspecting dump. Format per line: 
# doc, sum1, label1, sum2, label2, sum3, label3, ...

dump_format = "compact"

my_batch_size = 2**8  # too large or too small reduces GPU utility rate. 
# The speed is about 10 seconds per 2**8 doc-sum pairs on 3090

#========= NLP parameters

special_characters_to_clean = ['\n', '\t'] # replace such strings in raw data 

sent_end = [".", "!", "?"]  # symbols that represent the end of a sentence 
sent_end = string.punctuation

#========= negative sampling parameters 

# ratio between negative and positive samples
# minimal: 1 
neg_pos_ratio = 5

# methods used to generate negative samples 
methods = ["delete"] 
