#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_hub as hub


def sentence_embedding():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    # Compute a representation for each message, showing various lengths supported.
    word = "Elephant"
    sentence = "I am a sentence for which I would like to get its embedding."
    paragraph = (
        "Universal Sentence Encoder embeddings also support short paragraphs. "
        "There is no hard limit on how long the paragraph is. Roughly, the longer "
        "the more 'diluted' the embedding will be.")
    messages = [word, sentence, paragraph]

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(messages))

        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            print("Message: {}".format(messages[i]))
            print("Embedding size: {}".format(len(message_embedding)))
            message_embedding_snippet = ", ".join(
                (str(x) for x in message_embedding[:3]))
            print("Embedding: [{}, ...]\n".format(message_embedding_snippet))


class SentenceEmbedder():
    """This module is outputing:

    INFO:tensorflow:Saver not created because there are no variables
    in the graph to restore

    Quite annoying, so:

    >>> tf.logging.set_verbosity(logging.WARN)
    """
    def __init__(self):
        # 
        self.module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.embed_session = tf.Session()
        self.embed_session.run(tf.global_variables_initializer())
        self.embed_session.run(tf.tables_initializer())
        # self.module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    def embed(self, sentences):
        with tf.device('/cpu:0'):
            embedded = self.module(sentences)
            res = self.embed_session.run(embedded)
            return res
    def embed_list(self, sentences_list):
        with tf.device('/cpu:0'):
            embedded = [self.module(sentences) for sentences in sentences_list]
            res = self.embed_session.run(embedded)
            return res

def myembed(sentence):
    embedder = SentenceEmbedder()
    embedder.embed(sentence)
    """Embed a string into 512 dim vector
    """
    sentence = ["The quick brown fox jumps over the lazy dog."]
    sentence = ["The quick brown fox is a jumping dog."]
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    # embeddings = embed(["The quick brown fox jumps over the lazy dog."])
    embed_session = tf.Session()
    embed_session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    with tf.device('/cpu:0'):
        embedded = embed(sentence)
    res = embed_session.run(embedded)
    return res

def test():
    myembed(["The quick brown fox jumps over the lazy dog."])
    with tf.device('/cpu:0'):
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        embeddings = embed(["The quick brown fox jumps over the lazy dog."])
        session = tf.Session()
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedded = session.run(embeddings)
        print (embedded)
    pass

def main():
    with tf.device('/cpu:0'):
        sentence_embedding()
