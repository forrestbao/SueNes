#!/usr/bin/env python3

from preprocessing import preprocess_USE_story
from preprocessing import preprocess_InferSent_story
from preprocessing import preprocess_InferSent_negative
from preprocessing import preprocess_USE_Large_negative
from preprocessing import preprocess_USE_Large_story

if __name__ == '__main__':
    # preprocess_USE_Large_story(5)
    # preprocess_USE_Large_story(30)
    preprocess_USE_Large_story(100)
    # preprocess_USE_Large_negative(500)
    # preprocess_USE_Large_story(100)
    # preprocess_USE_Large_story(100)
    # preprocess_USE_Large_story(100)
    # preprocess_USE_Large_story(100)
    # preprocess_USE_Large_story(5000)

