# for testing sentence encoders on your GPU environment
#%%

import time, os 
import functools, multiprocessing, operator
import pickle, csv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow_hub as hub
import tensorflow as tf
import stanza 

# TODO: add document about the output format
# TODO: rectify output format be sequences of numpy.array. 

#%%
def embed_from_taskfile(taskfilename, dumpfile, sentence_splitter, sentence_encoder, batch_size=10):
    """load document-summary pairs with labels and incrementally convert sentences into vectors.

    input format: doc\tsum\tlabel 
    """

    try:
        os.remove(dumpfile)
    except OSError:
        pass

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
    # first split and get unique sentences 
    pairs, labels, unique_sentences = split_sentences(batch, sentence_splitter)

    # then encode all sentences 
    print ("embedding sentences", end="...", flush=True)
    sentence_vectors = sentence_encoder(unique_sentences)

    # third, break up
    print ("breaking things up...", end=" ", flush=True)
    triplets_in_vectors = \
    tuple( 
        (
            tf.gather(sentence_vectors, pair[0]), # doc 
            tf.gather(sentence_vectors, pair[1]), # sum 
            labels[pair_ID]                             # label 
        ) 
        for pair_ID, pair in enumerate(pairs)
    )

    print ("Dumping", end=" ", flush=True)
    # dump
    pickle.dump(triplets_in_vectors, dumpfile_handler)

    print ("done", end="\r", flush=True)

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
    print ("packing sentences", end="...", flush=True)
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
    print ("splitting sentences", end="...", flush=True)
    split_result = sentence_splitter(combined_paragraph)
    split_result = [sentence.text for sentence in split_result.sentences]
    # print (split_result)
    # Split result: 
    # DOC, s1, s2, s3, SUM, s4, s4, SUM, s5, s6, DOC, s7, s8, s9, SUM, s10, s11 END

    # third, process the splitted sentences. 
    # Create a long list of strings, unique_sentences, which are all unique sentences in this batch
    # return doc-sum pairs where sentneces are their indexes in unique_sentences 
    print ("indexing sentences", end="...", flush=True)
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
        
    return pairs, labels,  unique_sentences





#%%
def small_test():
    sentence_splitter, sentence_encoder =  init(tokenize_batch_size=128, sentence_encoder_name="google_use_large_5")
    taskfilename = "../data/cnn_dailymail_tiny/cross/train.tsv"
    dumpfilename = "../data/test_embedded.pickle"
    embed_from_taskfile(taskfilename, dumpfilename, sentence_splitter, sentence_encoder, batch_size = 128)

    # batch_size needs about 0.15x of memory in GiB. So 128 needs about 18GiB memory.


#%%

def init(tokenize_batch_size, sentence_encoder_name):
    # Currently we only use Stanza (Python version of Stanford Core NLP) to tokenize
    try: 
        sentence_splitter = stanza.Pipeline(lang='en', processors='tokenize', tokenize_batch_size=tokenize_batch_size)
    except Exception:
        stanza.download('en')
        sentence_splitter = stanza.Pipeline(lang='en', processors='tokenize', tokenize_batch_size=tokenize_batch_size)

    if "google_use" in sentence_encoder_name:
        # google_USE_location_prefix = "https://tfhub.dev/google/"
        google_USE_location_prefix = "/mnt/insecure/data/tfhub" 

        google_USE_location = google_USE_location_prefix + "/" + sentence_encoder_name

        sentence_encoder = hub.load(google_USE_location)

    return sentence_splitter, sentence_encoder

def loop_over():
    sentence_splitter, sentence_encoder = init(tokenize_batch_size=128, sentence_encoder_name="google_use_large_5")

    datasets = ["cnn_dailymail_tiny"]
    methods = ["cross"]
    splits = ["test"]

    # datasets = ["cnn_dailymail"]
    # methods = ["cross", "add", "delete", "replace"]
    # splits = ["train", "validate", "test"]

    taskfile_path_syntax = "'../data/'+dataset + '/' + method + '/'+split+'.tsv'" 
    dumpfile_path_syntax = "'../data/'+dataset + '/' + method + '/'+split + '.pickle'" 

    for dataset in datasets:
        for method in methods:
            for split in splits:
                taskfile_path = eval(taskfile_path_syntax)
                dumpfile_path = eval(dumpfile_path_syntax)
                embed_from_taskfile(taskfile_path, dumpfile_path, sentence_splitter, sentence_encoder, batch_size = 128)
    

if __name__ == "__main__":
   loop_over() 