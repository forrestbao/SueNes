# for testing sentence encoders on your GPU environment


import tensorflow_hub as hub

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]

# google encoders 
google_USE_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
google_USE_dan = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


google_large_embeddings = google_USE_large(sentences)
google_USE_embeddings = google_USE_dan(sentences)


