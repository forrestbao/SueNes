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

# DEPRECATED
EMBEDDING_DIM = 100

# VALIDATION_SPLIT = 0.2

DATA_DIR = "/home/hebi/mnt/data/nlp/"
# CNN_DIR = os.path.join(DATA_DIR, 'cnn/stroies')
# DM_DIR = os.path.join(DATA_DIR, 'dailymail/stories')
CNN_TOKENIZED_DIR = os.path.join(DATA_DIR, 'cnn_tokenized_stories')
DM_TOKENIZED_DIR = os.path.join(DATA_DIR, 'dailymail_tokenized_stories')


STORY_PICKLE_FILE = os.path.join(DATA_DIR, 'story.pickle')
WORD_MUTATED_FILE = os.path.join(DATA_DIR, 'word-mutated.pickle')
SENT_MUTATED_FILE = os.path.join(DATA_DIR, 'sent-mutated.pickle')
NEGATIVE_SAMPLING_FILE = os.path.join(DATA_DIR, 'negative-sampling.pickle')
NEGATIVE_SHUFFLE_FILE = os.path.join(DATA_DIR, 'negative-shuffle.pickle')

USE_DAN_DIR = os.path.join(DATA_DIR, 'USE-DAN')
USE_TRANSFORMER_DIR = os.path.join(DATA_DIR, 'USE-Transformer')

INFERSENT_DIR = os.path.join(DATA_DIR, 'InferSent')

INFERSENT_MODEL_PATH = os.path.join('/home/hebi/github/reading/InferSent/',
                                    'encoder/infersent2.pkl')
INFERSENT_W2V_PATH = os.path.join('/home/hebi/github/reading/InferSent/',
                                  'dataset/fastText/crawl-300d-2M.vec')

USE_BATCH_SIZE = 10240
USE_LARGE_BATCH_SIZE = 2560


# by default, it seems to use 2080ti alone, but I'd better make it
# explicit
#
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # 2080ti
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # 1070
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
