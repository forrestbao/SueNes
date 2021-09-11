import json, os, re
from summ_eval.supert_metric import SupertMetric
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.blanc_metric import BlancMetric

def main():
    # Fix SummaQA summa_qa_utils.py:20 code from huggingface see https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering
    scorers = [SupertMetric(), SummaQAMetric(), BlancMetric(inference_batch_size=32)]

    in_file = 'newsroom_60.tsv'

    docs = []
    sums = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            article, summary = line.split('\t')
            docs.append(article.strip())
            sums.append(summary.strip())
    
    for scorer in scorers:
        scores = scorer.evaluate_batch(sums, docs, aggregate=False)
        scorer_names = list(scores[0].keys())
        for scorer_name in scorer_names:
            with open(os.path.join("metric_"+scorer_name+".tsv"), "w", encoding="utf-8") as f:
                for score in scores:
                    f.write(str(score[scorer_name])+"\n")
            
if __name__ == '__main__':
    main()
