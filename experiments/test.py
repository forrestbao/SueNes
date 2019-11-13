import sys
sys.path.append("..")

from antirouge import preprocessing as pre

if __name__ == '__main__':
    #pre.preprocess_story_pickle()
    pre.preprocess_word_mutated()
    pre.preprocess_negative_sampling()