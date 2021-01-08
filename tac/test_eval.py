import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
import os, json, csv

article_per_set = 10
num_summarizer = 47

def read_tac_test_result(BERT_result_file, tac_json):
    """Load BERT result of using TAC2010 as test set and average over 10 articles

    By BERT convention, the file name is test_result.tsv 
    46 document sets, 47 summarizers (2 baselines, 41 machine ones, and 4 humans)

    one number per line 

    Due to how samples are generated, this is the correspondence between lines and doc-sum pairs
    docset_1: 
        article_1:
            summarizer_1, prediction 
            summarizer_2, prediction 
            ...
            summarizer_47, prediction 
        article_2: 
            summarizer_1, prediction 
            summarizer_2, prediction 
            ...
            summarizer_47, prediction 
        ...
        article_10: 
            ...
    docset_2: 
    ...
    docset_46: 

    what we want is:
    docset 1:
        summarizer_1, average of 10 predictions for article_{1,2,...,10}
        summarizer_2, average of 10 predictions for article_{1,2,...,10}
        ...
        summarizer_47, average of 10 predictions for article_{1,2,...,10}
    docset 2:
    ...
    docset_46: ...
    """

    with open(BERT_result_file, "r") as f:
        all_lines = f.read() 

    lines = all_lines.split("\n")
    if lines[-1] == "":
        lines = lines[:-1]  # get rid of last empty line 

    tac = json.load(open(tac_json, 'r'))
    score_dict = {}  # keys as (docset, summarizer), values as list of 10 floats
    docset_counter, article_counter, summarizer_counter = 0,0,0
    # Note that docset_counter, article_counter, nor summarizer_counters is actual docset, article or summarizer IDs. It's just counters to know whether we loop to next article. 
    
    for line in lines:
            # print (docset_counter, article_counter, summarizer_counter)
        docset = list(tac.keys())[docset_counter]
        summarizer = list(tac[docset]["summaries"].keys())[summarizer_counter]
        key = (docset, summarizer)
        score_dict.setdefault(key, []).append(float(line))
        
        if summarizer_counter == 47 - 1:
            summarizer_counter = 0
            if article_counter == 10 - 1: 
                article_counter = 0
                docset_counter += 1 
            else:
                article_counter += 1 
        else:
            summarizer_counter += 1 


    # Now, convert to the order in tac and get average 
    score_sorted = [] 
    for docset in tac.keys():
        for summarizer in tac[docset]["summaries"].keys():
            ten_scores = score_dict[(docset, summarizer)]
            avg_score = sum(ten_scores)/len(ten_scores)
            score_sorted.append(avg_score)

    return score_sorted

def load_tac_json(task_json):
    """Load the human scores from TAC from the JSON file compiled and dumped by our tac.py script 

    task_json: the JSON file containing all TAC samples and their human scores
    result_file: the test_result.txt by BERT on the test set

    The order of extracting scores from task_json needs to match that in _pop_tac_samples() in run_classifier.py


    order:
    docset_1, summarizer_1, scores[0:2]
    docset_1, summarizer_2, scores[0:2]
    ...
    docset_1, summarizer_47, scores[0:2]

    docset_2, summarizer_1, scores[0:2]
    docset_2, summarizer_2, scores[0:2]
    ...
    docset_2, summarizer_47, scores[0:2]
    ...
    ...
    ...
    docset_46 

    return: 
      dict: (docset_ID, summarizerID): scores[0:2]

    """

    tac_scores = [] # 46 x 47 rows, 3 columns

    tac = json.load(open(task_json, 'r'))
    for docset in tac.keys():
        tac_scores += [ tac[docset]["summaries"][summarizer]["scores"] for summarizer in tac[docset]["summaries"].keys() ]

    return tac_scores 

def calc_cc(tac_results, tac_scores):
    """Compute the correlation coefficients between BERT results on TAC test set and human evaluated scores on TAC test set

    tac_results: 1-D list of floats, 46(docset)x47(summarizers) elements
    tac_scores: 2-D list of floats, 46(docset)x47(summarizers) rows, and 3 columns
    """
    tac_scores = np.array(tac_scores)

    for i in range(3):
        corr_pearson = pearsonr(tac_results, tac_scores[:, i])
        corr_spearman = spearmanr(tac_results, tac_scores[:, i])

        print(corr_pearson[0], corr_spearman[0])
    print("---------------------------")


def cc_all():
    BERT_result_prefix = "/mnt/insecure/data/anti_rogue_result/cnn_dailymail/"
    tac_json_file = "./TAC2010_all.json"

    for method in ["cross", "add", "delete", "replace"]:
        print (method)
        BERT_result_file = os.path.join(BERT_result_prefix, method, "test_results.tsv")
        tac_results = read_tac_test_result(BERT_result_file, tac_json_file)
        tac_scores = load_tac_json(tac_json_file)
        calc_cc(tac_results, tac_scores)

if __name__ == "__main__":
    cc_all()
