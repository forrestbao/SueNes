# Preprocessing 

GNU GPL 3.0+
Forrest Sheng Bao 2020 


## Step 1: Sample generation 
Two ways to generate labled samples: cross pairing and mutation 

Specify the data and methods to generate samples in `sample_conf.py` 
Comments in the configuration file contains information about them.
If you are still confused, contact Forrest. 

The parameter that you might wanna change is `take_percent` which is 1 
when the code is commited. 
1 means only 1 percent of the data is used, for quick testing. 
Set it to a number such that your GPU can handle in reasonable amount
of GPU memory and time. 
If you are super rich, set it to 100! 

```shell
python3 sample_generation.py
```

By default, samples are dumped into TSV files of the naming convention
`dataset_GenerationMethod_Split`


## Step 2: Sentence embedding
 To do . 

## Next: Train the model. See model folder 
To do. 

