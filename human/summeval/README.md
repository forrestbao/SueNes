# Alignment with human evaluation in [SummEval dataset](https://github.com/Yale-LILY/SummEval)

1. Download the CNNDM dataset and human evaluation results from SummEval
```
mkdir -p cnndm

gdown "https://drive.google.com/u/0/uc?export=download&confirm=Fiu7&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"

gdown "https://drive.google.com/u/0/uc?export=download&confirm=0051&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs"

tar -xvzf cnn_stories.tgz -C cnndm

tar -xvzf dailymail_stories.tgz -C cnndm

wget -O model_annotations.aligned.scored.jsonl "https://drive.google.com/u/0/uc?id=1d2Iaz3jNraURP1i7CfTqPIj8REZMJ3tS&export=download"

python pair_data.py --data_annotations model_annotations.aligned.scored.jsonl --story_files .

rm -r cnndm/
```

2. Run ``generate_test.py`` to generate test file (``summeval_100.tsv``).

3. Run ``test_eval.py`` to compute the correlation between human evaluation scores and those from our model and baselines. 

