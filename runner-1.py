#!/usr/bin/env python3

from preprocessing import preprocess_sentence_embed
from main import create_or_load_data

if __name__ == '__main__':
    # preprocess_sentence_embed('USE-Large', 'mutated', 100, 10000)
    # preprocess_sentence_embed('USE', 'mutated', 5000, 10000)
    # preprocess_sentence_embed('InferSent', 'story', 300, 30000)
    create_or_load_data('neg', 'InferSent', 30000, 1)
