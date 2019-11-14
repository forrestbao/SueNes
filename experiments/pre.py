import sys
sys.path.append("..")

from antirouge import preprocessing as pre

def gen_dataset():
    pre.preprocess_story_pickle()
    pre.preprocess_word_mutated()
    pre.preprocess_negative_sampling()

if __name__ == '__main__':
    pre.preprocess_sentence_embed('USE', 'story', 1000, 40000, 0)