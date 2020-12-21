# for testing sentence encoders on your GPU environment
#%%

import time, os, sys
import functools, multiprocessing, operator
import pickle, csv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow as tf
import stanza 


# import nltk
# nltk.download('punkt')

#%%
def embed_from_taskfile(taskfilename, dumpfile, sentence_splitter, sentence_encoder, batch_size=10):
    """load document-summary pairs with labels and incrementally convert sentences into vectors.

    input format: doc\tsum1\tlabel1\tsum2\tlabel2 ... 
    """

    print ("Start")

    try: # because we use append mode to write results to batch incrementally
        os.remove(dumpfile)
    except OSError:
        pass

    if not os.path.exists(os.path.dirname(dumpfile)):
        try:
            os.makedirs(os.path.dirname(dumpfile))
        except OSError as exc: # Guard against race condition
            raise

    dumpfile_handler = open(dumpfile, 'ab')

    batch = []
    count = 0 
    global_counter = 0 
    with open(taskfilename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            if count >= batch_size:
                global_counter  += batch_size
                print ("at line %s" % global_counter, end='...')
                # return batch # debug 
                result = process_batch(batch, dumpfile_handler, sentence_splitter, sentence_encoder)
                count  = 0 
                batch = []
                # break # DEBUG
            else:
                batch.append(row)
                count += 1 

    if count>0: # last incomplete batch 
        print ("finishing last batch of %s lines" % len(batch))
        result = process_batch(batch, dumpfile_handler, sentence_splitter, sentence_encoder)

    dumpfile_handler.close()

    return batch

def process_batch(batch, dumpfile_handler, sentence_splitter, sentence_encoder):
    """
    """
    start = time.time()

    # first split and get unique sentences 
    pairs, labels, unique_sentences = split_sentences(batch, sentence_splitter)

    # then encode all sentences 
    print ("embedding", end="...", flush=True)

    sentence_vectors = sentence_encoder(unique_sentences)
    if tf.is_tensor(sentence_vectors): # for output from Google USE, convert to numpy array
        sentence_vectors = sentence_vectors.numpy()

    # third, break up
    # print ("breaking up...", end=" ", flush=True)
    triplets_in_vectors = \
    tuple (
        (   sentence_vectors[pair[0]], # doc
            sentence_vectors[pair[1]], # sum 
            labels[pair_ID]           # label 
        ) 
        for pair_ID, pair in enumerate(pairs)
    )

    print ("Dumping", end=" ", flush=True)
    # dump
    pickle.dump(triplets_in_vectors, dumpfile_handler)

    speed = len(batch)/(time.time() - start)
    print ("%f lines/s" % speed, end="\r", flush=True)

    return None

def split_sentences(batch, sentence_splitter):
    """

    >>> batch = [["I am happy. but i am sad.", "I am cool. i am confident. ", 0.5, "sunday is awesome. let's take a nap. ", 0.1], \
         [ "beef is ready. come over! ", "it is yummy. i want more. ", 0.8, "it's too spicy. pass me the yogurt. ", 0.3]]
    >>> stanza_splitter = stanza.Pipeline(lang='en', processors='tokenize')
    >>> pairs, labels, unique_sentences = split_sentences(batch, stanza_splitter)
    >>> pairs
    [([0], [1, 2]), ([0], [3, 4]), ([5, 6], [7, 8]), ([5, 6], [9, 10])]
    >>> labels
    [0.5, 0.1, 0.8, 0.3]
    >>> unique_sentences 
    ['I am happy. but i am sad.',
     'I am cool.',
     'i am confident.',
     'sunday is awesome.',
     "let's take a nap.",
     'beef is ready.',
     'come over!',
     'it is yummy.',
     'i want more.',
     "it's too spicy.",
     'pass me the yogurt.']

    Yes, Stanza did not split "I am happy. but i am sad." into two sentences. 
    """

    # first, pack all sentences, in documents or in summaries, into one long string. 
    # print ("packing", end="...", flush=True)
    combined_paragraph = "" # for parallel splitting 
    labels = [] 
    for line in batch: 
        _doc = " \n\n DOC \n\n " + line[0] 
        combined_paragraph += _doc 

        for i in range(1, len(line), 2):
            _sum = " \n\n SUM \n\n " + line[i]
            combined_paragraph += _sum 

        labels += [float(line[i]) for i in range(2, len(line), 2)]
    combined_paragraph += " \n\n THEEND \n\n "
    # print (combined_paragraph)
    
    # second, split. this is time consuming. 
    print ("splitting", end="...", flush=True)
    split_result = sentence_splitter(combined_paragraph)
    split_result = [sentence.text for sentence in split_result.sentences]
    # print (split_result)
    # Split result: 
    # DOC, s1, s2, s3, SUM, s4, s4, SUM, s5, s6, DOC, s7, s8, s9, SUM, s10, s11 END

    # third, process the splitted sentences. 
    # Create a long list of strings, unique_sentences, which are all unique sentences in this batch
    # return doc-sum pairs where sentneces are their indexes in unique_sentences 

    # print ("indexing", end="...", flush=True)
    unique_sentences = [] 
    sentence_ID = 0
    _sum = [] 
    _doc = []
    pairs = []
    is_doc = True # switch between sum and doc depending on whether DOC or SUM is scanned. 
    for sentence in split_result:
        # print (sentence, is_doc, _doc, _sum)
        if sentence == "DOC":
            if not is_doc : # was in summary mode 
                pairs.append( (_doc, _sum) ) # end of a summary
                _doc, _sum = [], [] # reset both
                is_doc = True 
        elif sentence == "SUM":
            if is_doc : 
                is_doc = False # switch to summary mode 
            else: # was in summary mode 
                pairs.append( (_doc, _sum) ) # end of a summary 
                _sum = [] # reset sum only
        elif sentence == "THEEND":
            pairs.append( (_doc, _sum) )
        else: 
            unique_sentences.append(sentence)
            if is_doc: 
                _doc.append(sentence_ID)
            else: 
                _sum.append(sentence_ID)
            sentence_ID += 1
        
    print ("%d sentences" % sentence_ID, end=" ", flush=True)
    return pairs, labels,  unique_sentences

#%%
def init(tokenize_batch_size, embedder_param:dict):
    # Currently we only use Stanza (Python version of Stanford Core NLP) to tokenize
    try: 
        sentence_splitter = stanza.Pipeline(lang='en', processors='tokenize', tokenize_batch_size=tokenize_batch_size)
    except Exception:
        stanza.download('en')
        sentence_splitter = stanza.Pipeline(lang='en', processors='tokenize', tokenize_batch_size=tokenize_batch_size)

    embedder_name = embedder_param["name"]

    if "google_use" in embedder_name.lower():
        import tensorflow_hub as tfhub
        google_USE_location_prefix = embedder_param.get("google_use_location_prefix", "https://tfhub.dev/google/")
        google_USE_location = google_USE_location_prefix + "/" + embedder_name
        sentence_encoder = tfhub.load(google_USE_location)

    elif "infersent" in embedder_name.lower():
        import torch
        sys.path.append(embedder_param["infersent_module_path"])
        from models import InferSent 
        infersent = InferSent(embedder_param)

        infersent.load_state_dict(torch.load(embedder_param['infersent_model_path']))
        infersent.set_w2v_path(embedder_param['word_vector_path'])

        infersent.build_vocab_k_words(K=100000) # load the K most common English words to save time 
        sentence_encoder = functools.partial(infersent.encode, 
                           tokenize=embedder_param['tokenization'])

    return sentence_splitter, sentence_encoder

def loop_over():
    ### Configure the sentence/word embedder 

    # If using Google USE, the location prefix is optional. 
    # If no local copies, default location https://tfhub.dev/google/ will be used 
    google_use_param = {"name":"google_use_large/5", 
                       # please use official USE names followed by a version number
                       # official USE names are given here https://tfhub.dev/google/collections/universal-sentence-encoder/1
                      "google_use_location_prefix": "/mnt/insecure/data/tfhub"}

    # InferSent v1 uses Glove, v2 uses FastText. They must match below. 
    # You need to manually add the InferSent Python module into os.path. so provide it below as well. 
    infersent_param = {"name":"infersent2", 
                      "infersent_module_path": "./InferSent", # where you cloned the InferSent repo from https://github.com/facebookresearch/InferSent
                      "infersent_model_path": "./InferSent/infersent2.pkl", # infersent v.2
                      "word_vector_path": "./InferSent/crawl-300d-2M.vec", # fasttext word vector, 
                      "tokenization":False, # set it to false to speed up using only split(). set to True to use NLTK word_tokenizer
                      # ALSO, modules for the infersent module below
                      'bsize': 64*4, # could be very big because we already batch the data loading 
                      'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2
                      }  

    ### Configure the I/O files 
    data_root_dir = '../data/'

    taskfile_path_syntax = "os.path.join('%s', dataset, method, split) + '.tsv'" % data_root_dir
    dumpfile_path_syntax = "os.path.join('%s', dataset, method, split, embedder_name) + '.pickle'" % data_root_dir

    ### Balance the memory and speed

    tokenize_batch_size = 128 # how many sentences to tokenize in a batch
    
    # how many lines from task file to load per batch
    # note that one line has one document and many candidate summaries. 
    # Memory needed is about 0.15x of lines in GiB, e.g, 128 lines => 18GiB RAM.
    # Google USE embeds all sentences passed to it at once. A big value here
    # will explode the GPU RAM. 
    # It has little effect on InferSent, which has a parameter bsize (above) to 
    # further sub-batch. 
    task_lines_per_batch = 128

    ### Configure the loop grid 

    ## a small test run
    embedder_params = [google_use_param]
    datasets = ["cnn_dailymail_tiny"]
    methods = ["delete"]
    splits = ["train"]

    ## A more realistic big run 
    # embedder_params = [google_use_param, infersent_param]
    # datasets = ["cnn_dailymail", "billsum", "scientific_papers"]
    # methods = ["cross", "add", "delete", "replace"]
    # splits = ["train", "validate", "test"]

    ### loop over all datasets, neg-sample methods, splits, and sentence embedders
    for dataset in datasets:
        for method in methods:
            for split in splits:
                for embedder_param in embedder_params: 
                    sentence_splitter, sentence_encoder = init(tokenize_batch_size, embedder_param)
                    embedder_name = embedder_param['name']
                    taskfile_path = eval(taskfile_path_syntax)
                    dumpfile_path = eval(dumpfile_path_syntax)
                    embed_from_taskfile(taskfile_path, dumpfile_path, sentence_splitter, sentence_encoder, batch_size = task_lines_per_batch)
    

if __name__ == "__main__":
   loop_over() 