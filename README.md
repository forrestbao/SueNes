# SueNes: A Weakly Supervised Approach to Evaluating Single-Document Summarization via Negative Sampling

A NAACL 2020 paper. 

## Dependencies and environment
* The negative sampling code requires TF 2.x and [`tensorflow_datasets`](https://www.tensorflow.org/datasets). 
* The `bert` code requires TF 1.15. We run our experiments using [nVidia's TF fork](https://github.com/NVIDIA/tensorflow). 
* SpaCy is needed for segmentation. 
* System: Ubuntu 20.04, 64GB RAM, RTX 3090


## To repeat our experiments

First, create folders under the directory of this project:

```bash
mkdir exp exp/data exp/result
```

### 1. Negative sampling
Code for generating negative samples are in `pre` folder. 

```bash
cd pre
python3 sentence_scramble.py # for sentence-level mutations 
python3 sample_generation.py # for crosspairing and word-level mutations
```
Configrations corresponding to the two Python scripts above are in  `sentence_conf.py` and `sample_generation.py`. Edit them to change negative sampling settings. 

### 2. Model training and test 

Code for model training and test is in the `bert` folder. 

Suppose now you are still in `pre` folder. 
```bash
cd ../bert # go one level up and then into the bert folder 
bash run_classifier.sh 
```

It will call our modified BERT's `run_classifier.py` script to train negative samples just generated above and to test on Newsroom, RealSumm, and TAC2010. Variable names in our `run_classifier.sh` bash script are made very self-explaintory for you to conveniently change the settings, such as the training set, test set, etc. 

Our `run_classifier.py` script hard-codes paths for the three test sets as: `./newsroom_60.tsv`, `./realsumm_100.tsv`, and `./TAC2010_all`. The files `newsroom_60.tsv` and `realsumm_100.tsv` are in this repo for convenience. TAC2010 is not because its access requires approval from NIST. All three files can be generated from raw data using scripts under `human` folder. Please refer to the README file under `human/{newsroom, realsumm, tac}` for information. 

### 3. Aligning with human evaluations 

Code for computing the correlation between our models' predictions and human ratings from the three datasets is in the `human` folder. 

## MISC 
Additional code are kept for reference, e.g., used in early stage of the development of our appproach:   
* `embed`: Scripts for sentence-level embedding. Kept for reference.
* `old`: Sentence-level models. Kept for reference. 

## Baselines and upperbounds

### Baselines: without using human-written reference summaries
* SUPERT: using heuristics to generate psedo-summaries
* BLANC: converting summary quality assessment into a question anwersing problem
* SummaQA: converting summary quality assessment into a question anwersing problem
* SUM-QE and WS_Score: problematic work 

### Upperbounds: using human-written reference summaries
