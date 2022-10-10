# Install virtual environment in Linux and install requirements
    - go to SueNes directory
    - python3 -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt
    - python -m spacy download en_core_web_sm
    - deactivate

# Part 1: download and generate training data
    - mkdir exp exp/data exp/result
    - cd pre
    - python3 sentence_scramble.py