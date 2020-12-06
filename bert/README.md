# BERT-related experiments 

Some important files under this folder are modified from [Google's official BERT release](https://github.com/google-research/bert).

Options for our script are very similar to those of BERT's officially released `run_classifier.py`. 

# Basic experiments, no TAC2010
Test sets are test sets of the original summarization datasets. 

## Crosspair-based

They are modeled as classification problems. So the Python script is `run_classifier.py` which is called in the script `run_classifier.sh` 

## Mutation-based

They are modeled as regression problems. So the Python script is `run_scorer.py` which is called in the script `run_scorer.py`

# Advanced experiments, with TAC2010
Test sets are from TAC2010 


# Files
* `data_processor.py`: Classes to load data in our own `plain` format where each row is `document \t summary \t label`

