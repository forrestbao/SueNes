# for testing sentence encoders on your GPU environment

import tensorflow_hub as hub
import tensorflow as tf
import time, os 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]* int(1*2**15)
DAN_scale_factor=4

print (tf.__version__)
TF_Version = int(tf.__version__[0])
#tf.debugging.set_log_device_placement(True)

if TF_Version == 1:
    google_USE_large = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    google_USE_dan = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    def embed_sents(sents, encoder):
      """Given a list of sentences, get their embeddings
      """
      config = tf.ConfigProto(allow_soft_placement = True)
      with tf.Session(config=config) as session:      
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(encoder(sents))  # numpy array
        return  (embeddings)

elif TF_Version == 2: 
        # google_USE_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        # google_USE_dan = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        google_USE_large = hub.load("/work/data/tfhub/google_use_large_5") 
        google_USE_dan = hub.load("/work/data/tfhub/google_use_dan_4")

#from tensorflow.python.client import device_lib
#for dev in device_lib.list_local_devices():
#    if dev.device_type in ["CPU", "GPU"]:
#        with tf.device(dev.name):
#print ("testing Google USE Transformer/Large on {}".format([dev.name]))

t=time.time()
print ("testing Google USE Transformer/Large")
if TF_Version == 1:
    embeddings = embed_sents(sentences, google_USE_large)
elif TF_Version == 2: 
    embeddings = google_USE_large(sentences)
print (embeddings) 
print ("elapsed seconds {}".format(time.time()-t))

t=time.time()
print ("testing Google USE DAN/small")
if TF_Version == 1:
    embeddings = embed_sents(sentences*DAN_scale_factor, google_USE_dan)
elif TF_Version == 2: 
    embeddings = google_USE_dan(sentences*DAN_scale_factor)
print (embeddings) 
print ("elapsed seconds {}".format(time.time()-t))

