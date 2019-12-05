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

ARTICLE_MAX_WORD = 1091
SUMMARY_MAX_WORD = 55

# CNN/DM
# ARTICLE_MAX_SENT = 47
# SUMMARY_MAX_SENT = 3

# DUC
# config.ARTICLE_MAX_SENT = 39
# config.SUMMARY_MAX_SENT = 5

ARTICLE_MAX_SENT = 50
SUMMARY_MAX_SENT = 5


# DEPRECATED
EMBEDDING_DIM = 100

# VALIDATION_SPLIT = 0.2

DATA_DIR = "F:\\Dataset\\CNN_DM"
# CNN_DIR = os.path.join(DATA_DIR, 'cnn/stroies')
# DM_DIR = os.path.join(DATA_DIR, 'dailymail/stories')
CNN_TOKENIZED_DIR = os.path.join(DATA_DIR, 'cnn_tokenized_stories')
DM_TOKENIZED_DIR = os.path.join(DATA_DIR, 'dailymail_tokenized_stories')


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

NEG_SIZE = 5

USE_DAN_DIR = os.path.join(DATA_DIR, 'USE-DAN')
USE_TRANSFORMER_DIR = os.path.join(DATA_DIR, 'USE-Transformer')

INFERSENT_DIR = os.path.join(DATA_DIR, 'InferSent')

INFERSENT_MODEL_PATH = os.path.join('/home/hebi/github/reading/InferSent/',
                                    'encoder/infersent2.pkl')
INFERSENT_W2V_PATH = os.path.join('/home/hebi/github/reading/InferSent/',
                                  'dataset/fastText/crawl-300d-2M.vec')

USE_BATCH_SIZE = 4096
USE_LARGE_BATCH_SIZE = 64
# this should be very small, and speed is still good
# INFERSENT_BATCH_SIZE = 64
INFERSENT_BATCH_SIZE = 32

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
