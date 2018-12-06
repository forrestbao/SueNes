#!/usr/bin/env python3

from preprocessing import preprocess_sentence_embed

if __name__ == '__main__':
    # preprocess_sentence_embed('InferSent', 'story', 10)
    # preprocess_sentence_embed('InferSent', 'negative', 10)
    preprocess_sentence_embed('USE-Large', 'story', 100)
    # preprocess_sentence_embed('USE', 'shuffle', 10000)
    # preprocess_sentence_embed('USE-Large', 'negative', 10)
    # preprocess_sentence_embed('InferSent', 'story', 10)


