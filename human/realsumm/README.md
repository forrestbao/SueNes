# Alignment with human evaluation in [RealSumm dataset](https://github.com/neulab/REALSumm)

1. Download the CNNDM dataset and human evaluation results from RealSumm
```
wget -O src.txt "https://drive.google.com/uc?export=download&id=1z1_i3cCQOd-1PWfaoFwO34YgCvdJemH7"

wget -O abs.pkl "https://github.com/neulab/REALSumm/blob/master/scores_dicts/abs.pkl?raw=true"

wget -O ext.pkl "https://github.com/neulab/REALSumm/blob/master/scores_dicts/ext.pkl?raw=true"
```

2. Run ``generate_test.py`` to generate test file (``realsumm_100.tsv``).

3. Run ``baselines.py`` to evaluate reference-based upperbounds.

4. Run ``ref_free_baselines.py`` to evaluate reference-free baselines.

## For our metric
1. See ``selected_docs_for_human_eval/selected_documents.md`` to download the dataset.
2. Run ``analysis/generate_test.py`` to generate test file (``test.tsv``) for our metric.

## For baselines
- Run ``analysis/baselines.py`` to evaluate reference-based upperbounds.
- Run ``analysis/ref_free_baselines.py`` to evaluate reference-free baselines.

## Analysis with human evaluation
1. Put all test results in ``predictions`` folder.
3. Run ``analysis/write_back.py`` to write the results to pkl file.
4. Run ``analysis/analysis.ipynb`` to do analysis.