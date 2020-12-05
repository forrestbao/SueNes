# Negative sampling 

## Dependencies 
* `tensorflow` 2.x (needed for tensorflow-datasets)
* `tensorflow-datasets` 

For example, you may install them using:

```sh
pip3 install tensorflow tensorflow-datasets 
```

## Usage 

Just run 
```bash 
python3 sample_generation.py
```
and have a cup of coffee or go to bed, and then data will be ready. 

Sample generation is controlled using a configuration file (such as `cnn_dailymail.py`), 
including specifying the dataset (e.g., `cnn_dailymail` or `big_patent`) and split (training, test, or validation) 
from which positive samples are extracted, the negative sampling method, etc. 
Such configurable settings are detailed in
comments of example configuration files. 
Some important ones are given below. 
The script `sample_generation.py` loads such configuration files for multiple single-document summarization datasets and generate samples accordingly. 

## Important configurations
* `load_percent`: Set it to 1 to use only 1 percent of data in a dataset for quick testing. Set it to 100 to use all -- slowest. Set it to an integer in between based on your computation power. 
* `dump_to`: Default paths of resulting files are `DATASET/METHOD/SPLIT.tsv`. For example, samples, both positive and negative, generated from the training set of CNN/DailyMail dataset using crosspairing, will be saved as `cnn_dailymail/cross/train.tsv`. You may change the path hierarchy and names here. 
* `dump_format`: Because for every positive document-summary pair, there could be multiple negative ones, we provide two output formats. When `plain` (default), samples are stored as a 3-column TSV file: 
    ``` 
    document \t summary \t label
    ``` 
  When `compact`, samples are stored as a multiple-column TSV file: 

    ``` 
    document \t 1st_summary \t 1st_label \t 2nd_summary \t 2nd_label ...
    ```




## Step 2: Sentence embedding
### Dependencies 
* tensorflow
* tensorflow\_hub
* pytorch 

### Step 2.1: Test whether Google USE and Facebook InferSent can run with your CPU and/or GPU on your system 
Run the two scripts separately: 

```sh
python3 google_USE_test.py
python3 InferSent_test.py
```
If you see a vector after every encoding task, then it runs correctly. 
Otherwise, email Forrest with all print-outs. 

## Next: Train the model. See model folder 
To do. 

