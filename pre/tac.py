# This script processes TAC results 

# Where are the files (using 2010 as example)
# 1. Documents: TAC2010_Summarizatioon_Documents.tgz 
#               -> GuidedSumm10_test_docs_files.tar.gz
#               -> 46 (NIST doc sometimes says 44) folders (e.g., D1022D)
#               -> two folders one for A and one for B (e.g., D1022D-B, D1022D-A)
# 2. Summaries: 
# 
#NIST assessors wrote 4 model summaries for each docset.
# The NIST human summarizer IDs are A-H.

#NIST received 41 runs from 23 participants for the guided
#summarization task.  The participants each submitted up to two runs,
#and their summarizer IDs are 3-43.

# Two baseline summarizer: Leadword (ID=1) and MEAD (ID=2)
# 3. Scores: 


import os, statistics
import bs4 # beautifulsoup
import numpy as np

# Human summarizer ID

def parse_tac_article(filename, sentence_delimiter):
    """Return a string where sentences are separated with _sentence_delimiter_
    """
#    print (filename)
    with open(filename) as f:
        s = bs4.BeautifulSoup(f, "html.parser")
        article = sentence_delimiter.join([p.get_text() for p in s.find_all("p")])
#        article = article.replace("\n", " ")

    return article 

def get_articles(dataset_path, setID, sentence_delimiter):
    """Extract articles from 2008-2010 updated/guided summary tasks 

    dataset_path: str, path to the parent directory of all docsets.
       Under the path, there are many folders, like D1022D, D1042H
        

    setID: list of str, consisting of "A" or "B"
        Most TAC summarization tasks uses two sets,
        each set consisting of 10 news articles

    sentence_delimiter: str, e.g., " ---- "

    return:
        dict, keys as document set (10 articles), values as list of 10 strings

    """
    articles = {} 
    for docset in os.listdir(dataset_path):
        for set_suffix in setID:
            docset_name = docset+"-"+set_suffix
            docset_path = os.path.join(dataset_path, docset, docset_name)
            for doc in os.listdir(docset_path):
                article = parse_tac_article(os.path.join(docset_path, doc), sentence_delimiter)
                articles.setdefault(docset_name, []).append(article)
    return articles 

def get_statistics(articles):
    """

    articles: dict, keys as document set (10 articles), values as list of 10 strings

    """
    c, w, s = [], [], [] # number of characters, words, and sentences
    for docset, _10_docs in articles.items():
        for doc in _10_docs:
            c.append(len(doc)) 
            w.append(len(doc.split(" ")))
            s.append(len(doc.split(". ")))

#    dist = [round(q, 1) for q in statistics.quantiles(lengths, n=10)] # need python3.8
    print ([int(np.percentile(c, x)) for x in range(1, 101, 10)])
    print ([int(np.percentile(w, x)) for x in range(1, 101, 10)])
    print ([int(np.percentile(s, x)) for x in range(1, 101, 10)])
    return None

def get_summaries():
    """Extract summaries from 2008-2011 updated/guided summary tasks 

    Two kinds of summaries: model summaries, 
       and automatic/machine generated summaries 

    """
    pass 


if __name__ == "__main__":
    dataset_path = "/mnt/insecure/data/TAC/TAC2010/TAC2010_Summarization_Documents/GuidedSumm10_test_docs_files/"
    setID = ["A"]
    sentence_delimiter = "  "
    articles = get_articles(dataset_path, ["A"], sentence_delimiter)
    print (get_statistics(articles))

