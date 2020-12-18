# for testing sentence encoders on your GPU environment
#%%

import time, os 
import functools, multiprocessing, operator
import pickle, csv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow_hub as hub
import tensorflow as tf
import stanza 


# google_USE_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# google_USE_dan = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# google_USE_large = hub.load("/home/forrest/tfhub/google_use_large_5") 

sentence_encoder_location = {
    "google_use_large": "/mnt/insecure/data/tfhub/google_use_large_5", 
    "google_use_dan": "/mnt/insecure/data/tfhub/google_use_dan_4"

}

try: 
    stanza_splitter = stanza.Pipeline(lang='en', processors='tokenize')
except Exception:
    stanza.download('en')
    stanza_splitter = stanza.Pipeline(lang='en', processors='tokenize')


"""
TODO: 
0. Use compact form of task file -- secondary 
1. assemble all docs and summaries, ending a doc with \n\n DOC \n\n and a summary with \n\n SUM \n\n
2. split the sentences. 
3. scan the sentence segmentation result. 
3.1 put every sentence into sentences_combined
3.2 turn doc_sum pairs into ids of sentences in setences_combined 
4. embed the sentences
5. break up and dump
"""

#%%
def split_sentence_pair(three_tuple):
    """Sentence segmentation. 

    TODO: how efficiently do this? We can combine all sentences in multiple docs and summaries together before sentence segmentation. But that may cause trouble when a document boundary is not detected as a sentence boundary. use a special character and then scan it. 
    """

    (_doc, _sum, label) = three_tuple
    doc_splitted = stanza_splitter(_doc)
    sum_splitted = stanza_splitter(_sum)
    return (
           tuple (sentence.text for sentence in doc_splitted.sentences) , 
           tuple (sentence.text for sentence in sum_splitted.sentences) , 
           label 
           )

#%%
def embed_from_taskfile(taskfilename, dumpfile, sentence_encoder, batch_size=10):
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
                print ("read line %s" % global_counter, end='...')
                result = process_batch(batch, dumpfile_handler, sentence_encoder)
                count  = 0 
                batch = []
                # break # DEBUG
            else:
                batch.append((row[0], row[1], float(row[2])))
                count += 1 

    if count>0: # last incomplete batch 
        print ("finishing last batch of %s lines" % len(batch))
        result = process_batch(batch, dumpfile_handler)

    dumpfile_handler.close()

    return batch

#%%
def turn_triplets_in_ID(triplets):
    """Re-represent the triplets using global IDs of sentences 

    global is within the batch 
    """

    def get_length_one_pair(three_tuple):
        return ( len(three_tuple[0]), len(three_tuple[1]) ) 

    number_of_sentences = map(get_length_one_pair, triplets)

    triplets_in_ID = [] 

    global_sentence_counter = 0 

    for pair_ID, (number_doc, number_sum) in enumerate(number_of_sentences):
        doc_IDs = tuple(range(global_sentence_counter, global_sentence_counter+number_doc)) 
        global_sentence_counter += number_doc
        sum_IDs = tuple(range(global_sentence_counter, global_sentence_counter+number_sum))
        global_sentence_counter += number_sum 
        label = triplets[pair_ID]
        
        triplets_in_ID.append((doc_IDs, sum_IDs, label))
    return triplets_in_ID
    # return map(get_length_one_pair, triplets)

def combine_sentences(triplets):
    """Pack all documents and summaries in a batch of (doc, sum, label) triplets into one tuple of sentences, each of which is a string. 

    add all tuples of sentences together 

    _doc, _sum: tuples of strings, each of which is a sentence 
    """

    return functools.reduce(operator.add,  
      ( _doc + _sum for (_doc, _sum, _) in triplets)
    ) # end reduce 

def process_batch(triplets, dumpfile_handler, sentence_encoder):
    """for each batch loaded from neg-sampled TSV file, split sentences and embed each
    """

    print ("splitting sentences...", end=" ", flush=True)
    triplets = tuple(map(split_sentence_pair, triplets))
    # FIXME: why do i have to manually turn it into a tuple?
    # TODO: use multiprocess to parallel it 
    
    print ("turning them into IDs", end=" ", flush=True)
    triplets_in_ID = turn_triplets_in_ID(triplets)

    # Pack all sentences together and send to embedder 
    print("packing all sentences...", end=" ", flush=True)
    sentences_combined = combine_sentences(triplets)
    print ("Encoding them ", end=" ", flush=True)
    sentence_vectors = sentence_encoder(sentences_combined)

    # break up 
    print ("breaking things up...", end=" ", flush=True)
    triplets_in_vectors = \
    tuple( 
        (
            tf.gather(sentence_vectors, three_tuple[0]), # doc 
            tf.gather(sentence_vectors, three_tuple[1]), # sum 
            three_tuple[2]                               # label 
        ) 
        for three_tuple in triplets_in_ID 
    )

    print ("Dumping", end=" ", flush=True)
    # dump
    pickle.dump(triplets_in_vectors, dumpfile_handler)

    print ("done", end="\r", flush=True)

    # return triplets, triplets_in_ID, sentences_combined, sentence_vectors, triplets_in_vectors
    return None


#%%


#embeddings = google_USE_large(sentences)

if __name__ == "__main__":
    sentence_encoder_name = "google_use_large"
    sentence_encoder = hub.load(sentence_encoder_location[sentence_encoder_name])
    taskfilename = "../data/test.tsv"
    dumpfilename = "../data/test_embedded.pickle"
    embed_from_taskfile(taskfilename, dumpfilename, sentence_encoder, batch_size = 200)