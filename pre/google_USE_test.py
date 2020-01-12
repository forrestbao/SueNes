# for testing sentence encoders on your GPU environment


import tensorflow_hub as hub
import tensorflow as tf

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]

# google encoders 


TF_Version = int(tf.__version__[0])

if TF_Version == 1:
    google_USE_large = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    google_USE_dan = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    def embed_sents(sents, encoder):
      """Given a list of sentences, get their embeddings
      """
      with tf.Session() as session:      
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(encoder(sents))  # numpy array
        return  (embeddings)

elif TF_Version == 2: 
    google_USE_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    google_USE_dan = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

from tensorflow.python.client import device_lib
for dev in device_lib.list_local_devices():
    if dev.device_type in ["CPU", "GPU"]:
        with tf.device(dev.name):
            print ("testing Google USE Transformer/Large on {}".format([dev.name]))
            if TF_Version == 1:
                embeddings = embed_sents(sentences, google_USE_large)
            elif TF_Version == 2: 
                embeddings = google_USE_large(sentences)
            print (embeddings) 

            print ("testing Google USE DAN/small on {}".format([dev.name]))
            if TF_Version == 1:
                embeddings = embed_sents(sentences, google_USE_dan)
            elif TF_Version == 2: 
                embeddings = google_USE_dan(sentences)
            print (embeddings) 


