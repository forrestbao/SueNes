# for testing sentence encoders on your GPU environment


import tensorflow_hub as hub
import tensorflow as tf

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]

# google encoders 
google_USE_large = hub.module("https://tfhub.dev/google/universal-sentence-encoder-large/5")
google_USE_dan = hub.module("https://tfhub.dev/google/universal-sentence-encoder/4")

from tensorflow.python.client import device_lib
for dev in device_lib.list_local_devices():
    if dev.device_type in ["CPU", "GPU"]:
        with tf.device(dev.name):
            print ("testing Google USE Transformer/Large on {}".format([dev.name]))
            google_large_embeddings = google_USE_large(sentences)
            print (google_large_embeddings)

            print ("testing Google USE DAN/small on {}".format([dev.name]))
            google_dan_embeddings = google_USE_dan(sentences)
            print (google_dan_embeddings)


