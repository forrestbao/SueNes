import sys
sys.path.append("..")
import tensorflow as tf
import pickle


from antirouge import preprocessing as pre
from antirouge import main
from antirouge import embedding


def tokenize():
    pre.preprocess_story_pickle(30000)
    pre.preprocess_word_mutated()
    pre.preprocess_negative_sampling()

def embed():
    #pre.preprocess_sentence_embed('USE', 'story', 30000, 30000)
    #pre.preprocess_sentence_embed('USE', 'negative', 30000, 30000)
    #pre.preprocess_sentence_embed('USE', 'mutated', 30000, 30000)
    #pre.preprocess_sentence_embed('USE-Large', 'story', 30000, 30000)
    #pre.preprocess_sentence_embed('USE-Large', 'negative', 30000, 30000)
    #pre.preprocess_sentence_embed('USE-Large', 'mutated', 30000, 30000)
    #pre.preprocess_sentence_embed('InferSent', 'story', 30000, 30000)
    #pre.preprocess_sentence_embed('InferSent', 'negative', 30000, 30000)
    pre.preprocess_sentence_embed('InferSent', 'mutated', 15000, 30000)
    pre.preprocess_sentence_embed('InferSent', 'mutated', 15000, 30000)
    

if __name__ == '__main__':
    # pre.preprocess_story_pickle(count_only = True)
    
    # tokenize()
    # embed()
    # main.run_exp2('neg', 'USE', 30000, 1, 'FC')
    '''
    answer = {}
    for embedding_method in ['USE', 'USE-Large', 'InferSent', 'glove']:
        answer[embedding_method] = {}
        for arch in ['CNN', 'FC', 'LSTM', '2-LSTM']:
            answer[embedding_method][arch] = {}
            for fake_option in ['add', 'delete', 'replace']:
                answer[embedding_method][arch][fake_option] = main.run_exp2('mutate', embedding_method, 30000, 1, arch, fake_option)
                print((embedding_method, arch, answer[embedding_method][arch][fake_option]))
    
    with open("result.pickle", 'wb') as f:
        pickle.dump(answer, f)
    '''
    
    answer2 = {}

    for embedding_method in ['USE', 'USE-Large', 'InferSent', 'glove']:
        answer2[embedding_method] = {}
        for arch in ['CNN', 'FC', 'LSTM', '2-LSTM']:
            answer2[embedding_method][arch] = main.run_exp2('neg', embedding_method, 30000, 1, arch)
            print((embedding_method, arch, answer2[embedding_method][arch]))
    
    with open("result.pickle", 'wb') as f:
        pickle.dump(answer2, f)
    
    print(answer, answer2)
    

    '''
    article_input = tf.keras.Input(shape=(None, 512), dtype='float32')
    summary_input = tf.keras.Input(shape=(None, 512), dtype='float32')
    x = tf.keras.layers.LSTM(128)(article_input)
    y = tf.keras.layers.LSTM(128)(summary_input)
    z = tf.keras.layers.Dense(1, activation='sigmoid')(tf.keras.layers.concatenate([x, y]))
    tf.keras.Model([article_input, summary_input], z)
    '''