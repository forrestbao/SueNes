# Anti-ROGUE

A supervised approach to summary quality assessment by negatively sampling on reference summaries 

## Organization of code

To repeat our experiments, run the code (with slight adaptation to your paths, datasets, etc.) in the following folders sequentialy. 
An end-to-end script (which will take really long to run and a substantial amounts of storage) will be provided shortly. 
1. `pre`: Scripts to generate negative samples from multiple public news datasets. 
2. `bert`: Modified BERT's `run_classifier.py` script to train on our 
  negatively sampled training data and to test on Newsroom, RealSumm, and TAC2010. To run the test, you need to extract human rated summaries first using code in the `human` folder first.
3.  `human`: Scripts related to extracting test data from Newsroom, RealSumm, and TAC2010, and computing the correlation between our models' predictions and human ratings from the three datasets. 

Additional code kept for reference, e.g., used in early stage of the development of our appproach:   
* `embed`: Scripts for sentence-level embedding. Kept for reference.
* `old`: Sentence-level models. Kept for reference. 

## Tensoflow versions
* The negative sampling code requires TF 2.x and `tensorflow_datasets`. 
* The `bert` code requires TF 1.15. We run our experiments using [nVidia's TF fork](https://github.com/NVIDIA/tensorflow). 