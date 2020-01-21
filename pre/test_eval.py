import numpy as np
import csv
from scipy.stats.stats import pearsonr, spearmanr

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

if __name__ == "__main__":
    test_file = "TAC2010_test.tsv"
    result_file = "test_results.tsv"

    test = read_tsv(test_file)
    result = read_tsv(result_file)
    
    num = len(test)
    article_per_set = 10
    feature_colmn = 1

    x = np.average(np.split(np.array([float(line[feature_colmn]) for line in result]), num / article_per_set), axis=1)
    y = np.array([[float(line[i]) for line in test] for i in range(2, 5)])
    y = y[:, ::article_per_set]

    for i in range(3):
        corr_pearson = pearsonr(x, y[i])
        corr_spearman = spearmanr(x, y[i])

        print(corr_pearson, corr_spearman)
                    
                