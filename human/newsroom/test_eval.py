# Collect and analyze results on newsroom dataset 

# Newsroom raw CSV foramt: 
# ArticleID,System,ArticleText,SystemSummary,ArticleTitle,CoherenceRating,FluencyRating,InformativenessRating,RelevanceRating

#%% 
import csv 
import scipy.stats
import os
import numpy 

def mean(a_list):
    # # print (a_list)
    # print (type(a_list))
    # print (type(a_list[0]))
    # print (sum(a_list))
    # return sum(a_list)/len(a_list)
    return numpy.mean(a_list)

def median(a_list):
    a_list.sort()
    # if len(a_list)!=3:
    #     print (a_list)
    return a_list[len(a_list)//2]

def load_newsroom_and_ours(newsroom_csv, prediction_tsv):
    """Load newsroom's human evaluation CSV file and run_classifier.py's prediction using our models. 
    The correspondence between documents and summaries in the prediction file is determined by to_tsv.py file. 

    Newsroom-csv is downloaded from
    https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv

    Return data structure: 
    {
        docID: {
            system1:{
                "Coherence":       float, 
                "Fluency":         float,
                "Informativeness": float,
                "Relevance":       float,
                "Ours":            float
            }
            system2: { ... }
            ...
            system7: {... }
        }
    }
    """

    predictions = [float(x.strip()) for x in open(prediction_tsv).readlines()]

    counter = -1 
    scores = {}  
    with open(newsroom_csv) as csvfile: 
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        for row in reader: 
            if counter > -1:
                docID, system = row[0:2]
                Coherence, Fluency, Informativeness, Relevance = list(map(float, row[5:9]))

                score_local = {
                            "Coherence" : [Coherence],
                            "Informativeness": [Informativeness],
                            "Fluency"  : [Fluency],
                            "Relevance": [Relevance]
                        }

                if docID not in scores: 
                    scores[docID] = {system: score_local 
                    }
                elif system not in scores[docID]:
                    scores[docID][system] = score_local 
                else: 
                    for aspect, score_list in score_local.items():
                        scores[docID][system][aspect] += score_list

                scores[docID][system]["Ours"] = [predictions[counter]]
            counter += 1

    return scores 

# Test 
# scores = load_newsroom_and_ours("./newsroom-human-eval.csv", "/mnt/12T/data/NLP/anti-rogue/result_base_sent/billsum/cross_sent_delete/test_results_newsroom.tsv")

# %% 
def summary_judge(scores, metrics_newsroom, metrics_other, consensus, correlation_types):
    """Summary-level evaluation between newsroom metrics and metrics from another group 

    metric_newsroom: ["Coherence", "Informativeness", "Fluency", "Relevance"]
    metric_other: ["Ours"]    # only ours for now. we can add others, e.g., ours from different training sets and/or negative sampling methods 

    concensus: string, "median", "mean", or "all" 

    """
    def get_correlation_per_doc_given_metrics(scores_of_one_document, metric_newsroom, metric_other, concensus, correlation_type):
        """

        scores of one document: 
        {
            system1:{
                "Coherence":       float, 
                "Fluency":         float,
                "Informativeness": float,
                "Relevance":       float,
                "Ours":            float
            }
            system2: { ... }
            ...
            system7: {... }
        }

        """
        vector_newsroom = [] # scores from a newsroom metric
        vector_other = []    # scores from a non-newsroom metric 
        for system, score_local in scores_of_one_document.items():
            score_newsroom = score_local[metric_newsroom] # list of 3 floats
            score_other = score_local[metric_other] # list of ONE float (TODO: need to expand to allow more than one float for multi-score metrics)
            if concensus in ["median", "mean", "max", "min"]:
                score_newsroom = eval(f"{concensus}(score_newsroom)")
                # if concensus == "median":
                #     score_newsroom.sort() 
                #     score_newsroom = score_newsroom[1] # type change 
                # elif concensus == "mean":
                #     score_newsroom = mean(score_newsroom) # type change 

                vector_newsroom.append(score_newsroom)
                vector_other   .append(score_other[0])
            elif consensus == "all":
                vector_newsroom += score_newsroom
                vector_other    += score_other*len(score_newsroom) # just duplicate 

        # print (vector_newsroom)
        # print ("asdfasdf")
        # print (vector_other)

        return  eval(f"scipy.stats.{correlation_type}(vector_newsroom, vector_other)")[0]
        # FIXME: This is a weird bug. It works only when manually convert from numpy.float64 to float. 
        # [0] to get the correlation. [1] is for pvalue. 

    # now begins the summary-level judge 
    correlations  = {}
    for correlation_type in correlation_types:
        # print (correlation_type)
        correlations[correlation_type] = {}
        for metric_newsroom in metrics_newsroom: # one metric from newsroom
            for metric_other in metrics_other:  # one metric to evaluate against newsroom 
                correlation_across_documents = [get_correlation_per_doc_given_metrics(scores_of_one_document, metric_newsroom, metric_other, consensus, correlation_type) for scores_of_one_document in scores.values()]
                # a list of floats

                correlations[correlation_type]\
                            [(metric_newsroom, metric_other)] = \
                            mean(correlation_across_documents)
    
    return correlations

# Test 
# correlations = summary_judge(scores, ["Coherence", "Informativeness", "Fluency", "Relevance"], ["Ours"], "mean")

def print_beautiful(correlation, correlation_types, metrics_newsroom):
    """
    correlation looks like this:
    {'kendalltau': {('Coherence', 'Ours'): 0.49600166520307726, ('Informativeness', 'Ours'): 0.5808293864832368, ('Fluency', 'Ours'): 0.45790855761575266, ('Relevance', 'Ours'): 0.4740934359726846}, 'spearmanr': {('Coherence', 'Ours'): 0.5906983676040027, ('Informativeness', 'Ours'): 0.6694288674637746, ('Fluency', 'Ours'): 0.5434449050990235, ('Relevance', 'Ours'): 0.5830224096341955}}
    """
    for correlation_type in correlation_types:
        for metric_newsroom in metrics_newsroom: 
            for metric_other in ["Ours"]:
                print ("{:1.4}".format(correlation[correlation_type][(metric_newsroom, metric_other)]), end="\t")

    print ("")

#%%
def main():

    # Configurations 
    concensus_based_on="median" # "median", "max", "min"
    metrics_newsroom = ["Coherence", "Informativeness", "Fluency", "Relevance"]
    short_correlation_types = ["p", "s", "k"]
    correlation_types = ["pearsonr", "kendalltau", "spearmanr"]

    # header to print 
    short_metrics_newsroom = [x[:3] for x in metrics_newsroom]
    cross_header = ["_".join([x,y]) for x in short_correlation_types for y in short_metrics_newsroom]
    print ("\t".join(["{:<17}".format("training_set"), " neg_sample"]+cross_header))

    
    data_root = "/mnt/12T/data/NLP/anti-rogue/result_base_sent"
    for training_set in ["billsum", "scientific_papers", "big_patent", "cnn_dailymail"]:
        for method in ["sent_delete", "sent_replace"]:
            print (f'{training_set:<17}\t', method, end="\t")
            prediction_tsv = os.path.join(data_root, training_set, method, "test_results_newsroom.tsv")
            scores = load_newsroom_and_ours("./newsroom-human-eval.csv", prediction_tsv)
            correlations = summary_judge(
                scores, 
                metrics_newsroom, 
                ["Ours"], 
                concensus_based_on, 
                correlation_types
                )
            print_beautiful(correlations, correlation_types, metrics_newsroom)

main()

def debug_for_hebi():
    # Configurations 
    concensus_based_on="median" # "median" or "median"
    metrics_newsroom = ["Coherence", "Informativeness", "Fluency", "Relevance"]
    correlation_types = ["pearsonr", "kendalltau", "spearmanr"]
    
    scores = load_newsroom_and_ours("./newsroom-human-eval.csv", "./prediction_billsum_sent_delete.tsv")
    correlations = summary_judge(
        scores, 
        metrics_newsroom, 
        ["Ours"], 
        concensus_based_on, 
        correlation_types
        )
    print_beautiful(correlations, correlation_types, metrics_newsroom)

# %%

def system_judge(scores):
    pass

def pooled_judge():
    pass


if __name__ == "__main__":
    debug_for_hebi()
# %%
