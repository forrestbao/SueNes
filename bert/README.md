# BERT-related experiments 

Some important files under this folder are modified from [Google's official BERT release](https://github.com/google-research/bert).

The Python script is `run_classifier.py` which is called in the script `run_classifier.sh`, where you can specify the type of experiments (dataset, augmentation method, and with or without human evaluation). Options for our script are very similar to those of BERT's officially released `run_classifier.py`. Following the tradition in GLUE tasks, each folder containing task data consists of three files, `train.tsv`, `validation.tsv`, and `test.tsv`. To automatically loop the experiments thru different datasets and different methods, please organize folders as 

```
      --data_dir=$DATA_DIR/$dataset/$method/
```

# Basic experiments, no human evaluation
Test sets are the test sets of original summarization datasets. Set 

```bash
exp_type=basic
```

# Advanced experiments with alignment to human evaluation 

To be implemented. 

Evaluation/development sets are the testsets of original summarization datasets. 
Test sets are from human evaluation data of TAC2010 and [Newsroom](https://github.com/lil-lab/newsroom/tree/master/humaneval). 



