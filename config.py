import os

# this seems to be not useful at all
MAX_NUM_WORDS = 200000

# np.median(words)            # 701
# np.median(sum_words)        # 46
# np.median(sents)            # 29
# np.median(sum_sents)        # 1
# np.percentile(words, 80)     # 1091
# np.percentile(sum_words, 80)  # 55
# np.percentile(sents, 80)      # 47
# np.percentile(sum_sents, 80)  # 1

# MAX_ARTICLE_LENGTH = 512
# MAX_SUMMARY_LENGTH = 128

# ARTICLE_MAX_SENT = 10
# SUMMARY_MAX_SENT = 3

MAX_ARTICLE_LENGTH = 1091
MAX_SUMMARY_LENGTH = 55

ARTICLE_MAX_SENT = 47
SUMMARY_MAX_SENT = 3

EMBEDDING_DIM = 100

# VALIDATION_SPLIT = 0.2

# CNNDM_DIR = '/home/hebi/github/reading/cnn-dailymail/'
CNNDM_DIR = '/home/hebi/mnt/data/cnn-dailymail/'
CNN_DIR = os.path.join(CNNDM_DIR, 'data/cnn/stroies')
DM_DIR = os.path.join(CNNDM_DIR, 'data/dailymail/stories')
CNN_TOKENIZED_DIR = os.path.join(CNNDM_DIR, 'cnn_stories_tokenized')
DM_TOKENIZED_DIR = os.path.join(CNNDM_DIR, 'dm_stories_tokenized')

STORY_PICKLE_FILE = os.path.join(CNNDM_DIR, 'story.pickle')
WORD_MUTATED_FILE = os.path.join(CNNDM_DIR, 'word-mutated.pickle')
SENT_MUTATED_FILE = os.path.join(CNNDM_DIR, 'sent-mutated.pickle')
NEGATIVE_SAMPLING_FILE = os.path.join(CNNDM_DIR, 'negative-sampling.pickle')
NEGATIVE_SHUFFLE_FILE = os.path.join(CNNDM_DIR, 'negative-shuffle.pickle')

USE_DAN_DIR = os.path.join(CNNDM_DIR, 'USE-DAN')
USE_TRANSFORMER_DIR = os.path.join(CNNDM_DIR, 'USE-Transformer')

INFERSENT_DIR = os.path.join(CNNDM_DIR, 'InferSent')

