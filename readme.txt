# Install virtual environment in Linux and install requirements
    - go to SueNes directory
    - python3 -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt
    - pip install pandas
    - pip install transformers
    - pip install datasets==1.0.2
    - rm seq2seq_trainer.py
    - wget https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/seq2seq_trainer.py
    - pip install rouge_score
    - python -m spacy download en_core_web_sm
    - export PYTHONPATH="${PYTHONPATH}:/home/jobayer/wsl-workspace/579X/SueNes/"
    - deactivate

# Part 1: download and generate training data
    - mkdir exp exp/data exp/result
    - cd pre
    - python3 sentence_scramble.py

