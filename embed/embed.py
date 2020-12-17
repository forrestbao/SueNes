# for testing sentence encoders on your GPU environment

import tensorflow_hub as hub
import tensorflow as tf
import stanza 

import time, os 
import csv 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]* int(1*2**15)

# google_USE_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# google_USE_dan = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
google_USE_large = hub.load("/home/forrest/tfhub/google_use_large_5") 
#        google_USE_dan = hub.load("/work/data/tfhub/google_use_dan_4")

def load_data(filename):
    """load document-summary pairs with labels 

    input format: doc\tsum\tlabel 
    """
    
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile, delimiter='\t')
        _doc_sum_label = [row for row in csvreader]

    return _doc_sum_label

def sentence_split(paragraph:str):
    """Sentence segmentation. 

    """
    
    try: 
        nlp = stanza.Pipeline(lang='en', processors='tokenize')
    except Exception:
        stanza.download('en')
        nlp = stanza.Pipeline(lang='en', processors='tokenize')
    doc = nlp(paragraph)
    return [sentence.text for sentence in doc.sentences] 
    

#embeddings = google_USE_large(sentences)

if __name__ == "__main__":
    pass