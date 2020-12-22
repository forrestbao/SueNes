# BERT-related experiments 

Some important files under this folder are modified from [Google's official BERT release](https://github.com/google-research/bert).

The Python script is `run_classifier.py` which is called in the script `run_classifier.sh`, where you can specify the type of experiments (dataset, augmentation method, and with or without human evaluation). Options for our script are very similar to those of BERT's officially released `run_classifier.py`. Following the tradition in GLUE tasks, each folder containing task data consists of three files, `train.tsv`, `validation.tsv`, and `test.tsv`, although the validation set is not used. `test.tsv` from the test set is used for validation. We use TAC2010 AESOP task as test set. So the 4th file called `TAC2010_all.json` must be placed properly if `--do-predict=True`. For information about `TAC2010_all.json`, please refer to [the TAC folder](../tac). 

To automatically loop the experiments thru different datasets and different methods, please organize folders as 

```
      --data_dir=$DATA_DIR/$dataset/$method/
```

If you augment the data using [our script](../pre/sample_generation.py), then the folders are already organized in that way. You just need to specify the correct `DATA_DIR`. 

# Dependencies

Tensorflow 1.x

It won't work for TF 2.x, although our data augmentation code is based on TF 2.x. 
