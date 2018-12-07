#!/usr/bin/env python3

from preprocessing import preprocess_sentence_embed

if __name__ == '__main__':
    # preprocess_sentence_embed('USE-Large', 'mutated', 100, 10000)
    # preprocess_sentence_embed('USE', 'mutated', 5000, 10000)
    preprocess_sentence_embed('InferSent', 'mutated', 300, 10000)
