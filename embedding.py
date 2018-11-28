#!/usr/bin/env python3


def embedder_test(embedder, articles):
    """Testing the performance of embedder.
    """
    # 7,742,028
    all_sents = []
    for article in articles:
        sents = sentence_split(article)
        all_sents.extend(sents)
        
    t = time.time()
    # 9.186472415924072
    s1000 = all_sents[:1000]
    embedder.embed(s1000)
    print(time.time() - t)
    
    t = time.time()
    # 11.613808870315552
    s5000 = all_sents[:5000]
    embedder.embed(s5000)
    print(time.time() - t)

    t = time.time()
    # 15.440065145492554
    s10000 = all_sents[:10000]
    embedder.embed(s10000)
    print(time.time() - t)

    t = time.time()
    # 49.14992094039917
    s50000 = all_sents[:50000]
    embedder.embed(s50000)
    print(time.time() - t)
    return

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
    def __init__(self):
        self.module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.embed_session = tf.Session()
        self.embed_session.run(tf.global_variables_initializer())
        self.embed_session.run(tf.tables_initializer())
    def embed(self, sentence):
        with tf.device('/cpu:0'):
            embedded = self.module(sentence)
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
