# Test on human evaluation from Newsroom dataset (https://github.com/lil-lab/newsroom)


## For our metric
1. Download the [human evaluation results](https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv)
2. Run ``to_tsv.py`` to generate test file (``newsroom_60.tsv``) for our metric.

## For baselines
- Run ``ref_free_baselines.py`` to evaluate reference-free baselines.

## Analysis with human evaluation
- ``test_eval.py``: computes the correlation for our model and other metrics to the human evaluation scores.