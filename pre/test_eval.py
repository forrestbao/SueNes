import numpy as np
import csv
from scipy.stats.stats import pearsonr, spearmanr
import os

article_per_set = 10

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def calc_cc(x, y):
    num = len(x)
    x = np.average(np.split(np.array(x), num / article_per_set), axis=1)
    y = np.array(y)
    y = y[:, ::article_per_set]

    for i in range(3):
        corr_pearson = pearsonr(x, y[i])
        corr_spearman = spearmanr(x, y[i])

        print(corr_pearson[0], corr_spearman[0])

if __name__ == "__main__":
    test_file = "TAC2010_test.tsv"
    result_path = "C:\\Users\\NKWBTB\\Desktop"
    #result_path = "C:\\Users\\NKWBTB\\Desktop\\Projects\\bert\\tmp\\bert_output_scorer_replace"
    #result_path = "C:\\Users\\NKWBTB\\Desktop\\Projects\\bert\\tmp\\bert_output_scorer_add"
    #result_path = "C:\\Users\\NKWBTB\\Desktop\\Projects\\bert\\tmp\\bert_output_scorer_delete"
    result_file = os.path.join(result_path, "test_results.tsv")
    
    feature_colmn = 0

    test = read_tsv(test_file)
    result = read_tsv(result_file)

    x_human, x_machine = [], []
    y_human, y_machine = [[],[],[]], [[],[],[]]
    for idx in range(len(result)):
        if int(test[idx][5]) == 0:
            x_human.append(float(result[idx][feature_colmn]))
            y_human[0].append(float(test[idx][2]))
            y_human[1].append(float(test[idx][3]))
            y_human[2].append(float(test[idx][4]))
        else:
            x_machine.append(float(result[idx][feature_colmn]))
            y_machine[0].append(float(test[idx][2]))
            y_machine[1].append(float(test[idx][3]))
            y_machine[2].append(float(test[idx][4]))

    print(len(x_human), len(y_human[0]))

    calc_cc(x_human, y_human)
    calc_cc(x_machine, y_machine)

                    
                