# Cropped from RealSumm (https://github.com/neulab/REALSumm)

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