import numpy as np
import json
import copy
import os
from scipy.stats.stats import pearsonr, spearmanr, kendalltau


def calc_corr(level, method, metric_scores, dataset, dataset_aspect):
    num_aspect = len(dataset_aspect)
    corr = []
    dataset_scores = []
    systems = 17
    docs = 100
    for aspect in dataset_aspect:
        scores = [np.mean([annotation[aspect] for annotation in example["expert_annotations"]]) for example in dataset]
        dataset_scores.append(scores) 
    
    dataset_scores = np.array(dataset_scores)
    metric_scores = np.array(metric_scores)
            
    if level == "pool":
        corr = [method(metric_scores, dataset[i])[0] for i in range(num_aspect)]
    elif level == "summary":
        dataset_scores = dataset_scores.reshape(num_aspect, docs, systems)
        metric_scores = metric_scores.reshape(docs, systems)
        corr = np.zeros((num_aspect, ), dtype=np.float32)
        for doc in range(docs):
            for i in range(num_aspect):
                if np.max(dataset_scores[i, doc, :]) == np.min(dataset_scores[i, doc, :]):
                    # A workaround to avoid constant vector issue when calculating correlation
                    dataset_scores[i, doc, 0] += 0.001
                if np.max(metric_scores[doc, :]) == np.min(metric_scores[doc, :]):
                    metric_scores[doc, 0] += 0.001
            corr += np.array([method(metric_scores[doc, :], dataset_scores[i, doc, :])[0] for i in range(num_aspect)])
        corr /= doc
    elif level == 'system':
        dataset_scores = dataset_scores.reshape(num_aspect, docs, systems)
        metric_scores = metric_scores.reshape(docs, systems)

        dataset_scores_system = np.mean(dataset_scores, axis=1)
        metric_scores_system = np.mean(metric_scores, axis=0)
        corr = np.array([method(metric_scores_system, dataset_scores_system[i])[0] for i in range(num_aspect)])
    else:
        print("???")
    
    line = ["%.4f"%i for i in corr]
    line = "\t".join(line)
    print (line, end = ' ')


def main():
    # Configurations 
    result_root = "../../exp/result_bert_base_uncased"
    training_sets = os.listdir(result_root)
    level="summary"

    dataset_aspect = ["coherence", "consistency", "fluency", "relevance"]
    dataset = []
    with open("model_annotations.aligned.scored.jsonl", "r", encoding="utf-8") as fd:
        dataset = [json.loads(line) for line in fd]
    
    for training_set in training_sets:
        methods = os.listdir(os.path.join(result_root, training_set))
        for method in methods:
            prediction_tsv = os.path.join(result_root, training_set, method, "test_results_summeval.tsv")
            if not os.path.exists(prediction_tsv):
                continue
            print (f'{training_set}-{method}', end="\t")

            with open(prediction_tsv, "r") as f:
                metric_scores = [float(line) for line in f]

        for method in [pearsonr, spearmanr, kendalltau]:
            calc_corr(level, method, metric_scores, dataset, dataset_aspect)

        print("")


    # Baselines
    sorted_keys = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score', 
        's3_pyr', 's3_resp', 'bert_score_precision', 'bert_score_recall', 
        'bert_score_f1', 'mover_score', 'summaqa_avg_fscore', 
        'blanc', 'supert', 'bleu', 'cider', 'meteor']
    
    for metric in sorted_keys:
        '''
        print(tsvfile, end = "\t")
        metric_scores = []
        with open(os.path.join(pred_path, tsvfile), "r") as f:
            metric_scores = [float(line) for line in f]
        '''
        print(metric, end = "\t")
        metric_scores = []
        for example in dataset:
            if metric.startswith("rouge"):
                metric_scores.append(example["metric_scores_11"]["rouge"][metric])
            elif metric == "supert":
                metric_scores.append(example["metric_scores_1"][metric][0])
            elif metric == "summaqa_avg_fscore" or metric == "blanc":
                metric_scores.append(example["metric_scores_1"][metric])
            else:
                metric_scores.append(example["metric_scores_11"][metric])
        
        for method in [pearsonr, spearmanr, kendalltau]:
            calc_corr(level, method, metric_scores, dataset, dataset_aspect)
            
        print("")



if __name__ == '__main__':
    main()