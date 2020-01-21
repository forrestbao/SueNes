from tac import *
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr

if __name__ == "__main__":
    article_set_path = "F:\\Dataset\\TAC2010/TAC2010/TAC2010_Summarization_Documents/GuidedSumm10_test_docs_files/"
    summary_set_path = "F:\\Dataset\\TAC2010/TAC2010/GuidedSumm2010_eval/ROUGE"
    score_path = "F:\\Dataset\\TAC2010/TAC2010/GuidedSumm2010_eval/manual"

    rouge_score_path = "F:\\Dataset\\TAC2010/TAC2010/GuidedSumm2010_eval/ROUGE/rouge_A.m.out"
    output_file = "rouge_score.tsv"

    setIDs = ["A"]
    sentence_delimiter = "  "
    summary_types = ["peers", "models"]

    hscore_type = 3
    rscore_type = 21
    
    
    #articles = get_articles(article_set_path, setIDs, sentence_delimiter)
    #_,_,_ = get_statistics(articles)

    #summaries = get_summaries(summary_set_path, setIDs, sentence_delimiter, summary_types)
                                                # sentence_delimiter,  NOT IN USE 

    scores = get_scores(score_path, summary_types, setIDs)
    print(scores.keys())
    # combined = dump_data(articles, summaries, scores, dump_to=dump_to)

    rouge_scores = get_rouge(rouge_score_path)

    x = [ [rouge_scores[key][i] for key in rouge_scores.keys()] for i in range(rscore_type)]
    y = [ [scores[doc][summarizer][i] for doc, summarizer in rouge_scores] for i in range(hscore_type)]

    corr_pearson = np.zeros((rscore_type, hscore_type), dtype=np.float32)
    corr_spearman = np.zeros((rscore_type, hscore_type), dtype=np.float32)
    for i in range(rscore_type):
        for j in range(hscore_type):
            corr_pearson[i, j] = pearsonr(x[i], y[j])[0]
            corr_spearman[i, j] = spearmanr(x[i], y[j])[0]

    output = np.concatenate((corr_pearson, corr_spearman), axis=1)

    with open(output_file, "w", encoding="UTF-8") as f:
        for line in output:
            strline = "\t".join([str(val) for val in line]) + "\n"
            f.write(strline)


