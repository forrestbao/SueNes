import sys
sys.path.append("..")

from antirouge import preprocessing as pre
from antirouge import main

def tokenize():
    pre.preprocess_story_pickle()
    pre.preprocess_word_mutated()
    pre.preprocess_negative_sampling()

def embed():
    pre.preprocess_sentence_embed('USE', 'story', 10000, 40000)
    pre.preprocess_sentence_embed('USE', 'negative', 10000, 40000)
    pre.preprocess_sentence_embed('USE', 'mutated', 10000, 40000)
if __name__ == '__main__':
    # embed()
    main.run_exp('neg', 'USE', 10000, 1, 'CNN')