# For details, see here https://github.com/facebookresearch/InferSent

# Jump to "here is the real beef" part for InferSent code 

# ================= house keeping code   ======
import os, sys

fasttext_model = "InferSent/crawl-300d-2M.vec"
infersent_model = "InferSent/infersent2.pkl"

try: # if you have no numpy installed or cfg'ed properlly, install using system package manager; right, i don't use pip to install them.
    import numpy
except ModuleNotFoundError:
    os.system("sudo apt install python3-numpy")

try: # if you have no nltk installed or cfg'ed properlly, install using system package manager; right, i don't use pip to install them.
    import nltk
except ModuleNotFoundError:
    os.system("sudo apt install python3-nltk python3-numpy-")

# clone infersent
if not os.path.isdir("InferSent"):
    os.system("git clone https://github.com/facebookresearch/InferSent.git --depth=1")


# download infersent model
if not os.path.isfile(infersent_model):
    os.system("curl -Lo " + infersent_model  + " https://dl.fbaipublicfiles.com/infersent/infersent2.pkl")

# download fasttext model
if not os.path.isfile(fasttext_model):
    os.system("curl -Lo " + fasttext_model  + ".zip  https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
    os.system("unzip InferSent/crawl-300d-2M.vec.zip -d InferSent") 
    # result in a file crawl-300d-2M.vec


# setup NLTK
import nltk
nltk.download('punkt')



#============ here is the real beef ======


# load and cfg infersent
sys.path.append("InferSent/")
from models import InferSent
import torch

params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}


def test_infersent(infersent):
    """
         _infersent_ is an instance of InferSent/models/InferSent.py
    """

    infersent.load_state_dict(torch.load(infersent_model))
    infersent.set_w2v_path(fasttext_model)

    # build vocabulary and encode the sentence
    sentences = ([
        "The quick brown fox jumps over the lazy dog.",
        "I am a sentence for which I would like to get its embedding"])

    infersent.build_vocab(sentences, tokenize=True)
    embeddings = infersent.encode(sentences, tokenize=True)

    print ("using InferSent v2 to encode sentences..")
    print (embeddings)


# test on CPU
infersent = InferSent(params_model)
print ("test on CPU")
test_infersent(infersent)

# test on GPU
infersent = InferSent(params_model)
infersent.cuda() # move to GPU
print ("test on GPU")
test_infersent(infersent)

