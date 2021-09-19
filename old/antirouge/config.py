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

# ARTICLE_MAX_WORD = 1091
# SUMMARY_MAX_WORD = 55

# CNN/DM 92579
# ARTICLE_MAX_WORD = 1091
# SUMMARY_MAX_WORD = 55
# ARTICLE_MAX_SENT = 47
# SUMMARY_MAX_SENT = 1

# News Room 995041
# ARTICLE_MAX_WORD = 1079
# SUMMARY_MAX_WORD = 36
# ARTICLE_MAX_SENT = 45
# SUMMARY_MAX_SENT = 2

# bigPatent 1207222
# ARTICLE_MAX_WORD = 4750
# SUMMARY_MAX_WORD = 156
# ARTICLE_MAX_SENT = 185
# SUMMARY_MAX_SENT = 5

# NYT 654873
ARTICLE_MAX_WORD = 1219
SUMMARY_MAX_WORD = 67
# ARTICLE_MAX_SENT = 51
# SUMMARY_MAX_SENT = 1


# DUC
# config.ARTICLE_MAX_SENT = 39
# config.SUMMARY_MAX_SENT = 5

ARTICLE_MAX_SENT = 50
SUMMARY_MAX_SENT = 5


# DEPRECATED
EMBEDDING_DIM = 100

# VALIDATION_SPLIT = 0.2

# CAUTION this folder is important, all raw data is expected to put here, and
# all generated preprocessed data will be put here as well

# DATA_DIR = "F:\\Dataset\\nyt_corpus"
DATA_DIR = os.path.expanduser("~/data")
CORENLP_JAR = os.path.join(DATA_DIR, 'stanford-corenlp-3.9.2.jar')

CNN_DIR = os.path.join(DATA_DIR, 'cnn/stories')
DM_DIR = os.path.join(DATA_DIR, 'dailymail/stories')
CNN_TOKENIZED_DIR = os.path.join(DATA_DIR, 'tokenized_stories')
# CNN_TOKENIZED_DIR = os.path.join(DATA_DIR, 'cnn_tokenized_stories')
# DM_TOKENIZED_DIR = os.path.join(DATA_DIR, 'dailymail_tokenized_stories')

KEYS_FILE = os.path.join(DATA_DIR, 'keys.pickle')
SHUFFLE_FILE = os.path.join(DATA_DIR, 'shuffle.pickle')

STORY_PICKLE_FILE = os.path.join(DATA_DIR, 'story.pickle')
WORD_MUTATED_FILE = os.path.join(DATA_DIR, 'word-mutated.pickle')
SENT_MUTATED_FILE = os.path.join(DATA_DIR, 'sent-mutated.pickle')
NEGATIVE_SAMPLING_FILE = os.path.join(DATA_DIR, 'negative-sampling.pickle')
NEGATIVE_SHUFFLE_FILE = os.path.join(DATA_DIR, 'negative-shuffle.pickle')

PROTO_DIR = os.path.join(DATA_DIR, 'proto')
# DEPRECATED
SERIAL_DIR = os.path.join(DATA_DIR, 'serial')
CNN_SERIAL_DIR = os.path.join(DATA_DIR, 'serial_cnn')
DM_SERIAL_DIR = os.path.join(DATA_DIR, 'serial_dm')

# probably just use 1 so that the positive and negative samples are balanced
NEG_SIZE = 1

USE_DAN_DIR = os.path.join(DATA_DIR, 'USE-DAN')
USE_TRANSFORMER_DIR = os.path.join(DATA_DIR, 'USE-Transformer')

INFERSENT_DIR = os.path.join(DATA_DIR, 'InferSent')

USE_BATCH_SIZE = 4096
USE_LARGE_BATCH_SIZE = 8
# this should be very small, and speed is still good
# INFERSENT_BATCH_SIZE = 64
INFERSENT_BATCH_SIZE = 4

# by default, it seems to use 2080ti alone, but I'd better make it
# explicit
#
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # 2080ti
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # 1070
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# will chunk data tfrec file every 10000 stories
DATA_BATCH_SIZE = 10000

DUC_2002_RAW_DIR = os.path.join(DATA_DIR, 'DUC2002')
DUC_2002_DIR = os.path.join(DATA_DIR, 'DUC2002_OUT')
