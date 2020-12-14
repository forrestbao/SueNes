#!/usr/bin/env python3

def test():
    with open('log.txt') as f:
        for line in f:
            if line.startswith('SETTING:'):
                print(line)
            if line.startswith('RESULT:'):
                print(line)
