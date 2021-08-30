from tac import *
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
import json

def calc_corr(metric_score, scores, rscore_type, hscore_type):
    x = [ [metric_score[key][i] for key in rouge_scores.keys()] for i in range(rscore_type)]
    y = [ [scores[doc][summarizer][i] for doc, summarizer in rouge_scores.keys()] for i in range(hscore_type)]

    corr_pearson = np.zeros((rscore_type, hscore_type), dtype=np.float32)
    corr_spearman = np.zeros((rscore_type, hscore_type), dtype=np.float32)
    for i in range(rscore_type):
        for j in range(hscore_type):
            corr_pearson[i, j] = pearsonr(x[i], y[j])[0]
            corr_spearman[i, j] = spearmanr(x[i], y[j])[0]

    output = np.concatenate((corr_pearson, corr_spearman), axis=1)
    return output

if __name__ == "__main__":  
    TAC_result_root = "/home/nkwbtb/Downloads/TAC2010"
    score_path = TAC_result_root + "/GuidedSumm2010_eval/manual"
    rouge_score_path = TAC_result_root + "/GuidedSumm2010_eval/ROUGE/rouge_A.m.out"

    output_file = "rouge_score.tsv"

    setIDs = ["A"]
    sentence_delimiter = "  "
    summary_types = ["peers", "models"]

    hscore_type = 3
    rscore_type = 21
    
    scores = get_scores(score_path, summary_types, setIDs)
    rouge_scores = get_rouge(rouge_score_path)
    
    output = calc_corr(rouge_scores, scores, rscore_type, hscore_type)

    with open(output_file, "w", encoding="UTF-8") as f:
        for line in output:
            strline = "\t".join([str(val) for val in line]) + "\n"
            f.write(strline)

    # Correlation for other baselines
    baselines_score_path = "baselines.json"
    baselines_score = {}
    with open(baselines_score_path, "r", encoding="utf-8") as f:
        baselines_score = json.load(f)

    akey = list(baselines_score.keys())[0]
    score_type = list(baselines_score[akey].keys())
    rscore_type = len(score_type)

    converted_score = {}
    for doc_file in baselines_score:
        score_list = []
        for score in score_type:
            score_list.append(baselines_score[doc_file][score])

        doc_string = doc_file.split('.')
        doc = doc_string[0]
        summarizer = doc_string[-1]
        converted_score[(doc, summarizer)] = score_list
    
    output = calc_corr(converted_score, scores, rscore_type, hscore_type)
    with open("baselines_score.tsv", "w", encoding="UTF-8") as f:
        lc = 0
        for line in output:
            strline = score_type[lc] + "\t" + "\t".join([str(val) for val in line]) + "\n"
            f.write(strline)
            lc += 1