# Embedding sentences for non-BERT baselines 

The script `embed.py` embeds sentences into vectors using Google Universal Sentence Encoder (DAN and transformer-based), Facebook InferSent, and plain word embedding (word2vec or GloVe). 

The intput format is in the `compact` format generated using [../pre/sample_generation.py](../pre/sample_generation.py) that each row is a document, followed by all summaries (positive or negatively sampled) and their labels. That is 
```
document \t summary 1 \t label 1 \t summary 2 \t label 2 ...
```
The number of columns varies. 

The output embedded document-summary pairs and their labels are organized as a list and dumped into a pickle file. 
In the list, each element is a 3-tuple, corresponding to the document, the summary, and the label. 
The document or the summary is a numpy array, whose rows correspond to sentences (if using sentence embedding) or words (if using word embedding). 

The dump file's name follow this path convention: `{data_root_dir}/{dataset}/{method}/{split}/{embedder}.pickle`, e.g., `../data/cnn_dailymail/replace/validation/google_use_large_5.pickle`. 

Parameters, including locations of embedders, are explained in the function `loop_over`. Just change variables in `loop_over` and call it under `__main__` for your own problems. 

# Depencencies
1. Stanza: for sentence segmentation  
2. NLTK: needed by InferSent
3. Torch: needed by InferSent

If you use InferSent, please clone InferSent and download model files (both InferSent's model and the word vector model it relies on). Then configure their locations in `infersent_param` in `loop_over` in `embed.py`. 

If you use one of Google Universal Sentence Encoders, you can either specify the local location of them, or use online URL (default, or when local location is not given), e.g., https://tfhub.dev/google/universal-sentence-encoder-lite/2 . Using local locations avoids download the models again and again. The `name` in `google_use_param` (in function `loop_over` in `embed.py`) needs to be an official embedder name (as given in [this page](https://tfhub.dev/google/collections/universal-sentence-encoder/1)) followed by a version number, e.g., `universal-sentence-encoder-lite/2`. So please organize your local file hierarchy accordingly. 