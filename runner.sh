#!/bin/bash

for (( i=0 ; i<20 ; i++ )); do
    # 500 negative
    python3 runner-2.py
    # 100 story
    python3 runner-1.py
    python3 runner-1.py
    python3 runner-1.py
    python3 runner-1.py
    python3 runner-1.py
done

