#!/usr/bin/env python3


# I'm not using ABSL becasue 1. it introduces global variables. That
# shares namespaces with all libraries. 2. it forces the code to run
# in command line. There is no way to define FLAGS.XXX in REPL, but
# has to have --xxx in argv string.

from absl import app
from absl import flags

import os

from antirouge import config
from antirouge import newmain
from antirouge import duc_tac_main
from antirouge import serial_preprocess

import argparse

def main1():
    parser = argparse.ArgumentParser(description='Anti rouge tester entry.')

    parser.add_argument('expname', choices=['USE', 'USE-Large',
                                            'glove', 'InferSent'])
    parser.add_argument('--neg-size', type=int)
    
    # args = parser.parse_args(['--neg-size', '5'])
    # args = parser.parse_args(['USE'])
    args = parser.parse_args()

    if args.neg_size is not None:
        config.NEG_SIZE = args.neg_size
    if args.expname == 'USE':
        newmain.exp_USE()
    elif args.expname == 'USE-Large':
        newmain.exp_USE_Large()
    elif args.expname == 'glove':
        newmain.exp_word()
    else:
        newmain.exp_InferSent()

def main2():
    duc_tac_main.exp()

def preprocess():
    parser = argparse.ArgumentParser(description='Anti rouge tester entry.')
    parser.add_argument('embedder', choices=['story', 'USE', 'USE-Large',
                                            'InferSent'])
    args = parser.parse_args()
    if args.embedder == 'story':
        serial_preprocess.serial_process_story(config.CNN_TOKENIZED_DIR,
                             os.path.join(config.CNN_SERIAL_DIR, 'story'))
    elif args.embedder == 'USE':
        serial_preprocess.serial_process_embed(config.CNN_SERIAL_DIR, 'USE', 3)
    elif args.embedder == 'USE-Large':
        serial_preprocess.serial_process_embed(config.CNN_SERIAL_DIR, 'USE-Large', 1)
    else:
        serial_preprocess.serial_process_embed(config.CNN_SERIAL_DIR, 'InferSent')
    

if __name__ == '__main__':
    main1()
    # main2()
    # preprocess()

# if __name__ == '__main__':
#   app.run(newmain.exp_USE)
