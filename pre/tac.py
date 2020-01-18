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





import os, statistics, glob
import bs4 # beautifulsoup
import numpy as np

# Human summarizer ID

def parse_tac_article(filename, sentence_delimiter):
    """Return a string where sentences are separated with _sentence_delimiter_

    In the source format, each <p> tag is one sentence. 
    
    
    TODO: 
    Forrest: Roger, please verify that is one sentence per <p> tag. 

    """
#    print (filename)
    with open(filename) as f:
        s = bs4.BeautifulSoup(f, "html.parser")
        article = sentence_delimiter.join([p.get_text() for p in s.find_all("p")])
#        article = sentence_delimiter.join([p.get_text() for p in s.find_all("p")])
        article = [p.get_text() for p in s.find_all("p")]

#        article = article.replace("\n", " ")

    return article 

def get_articles(dataset_path, setIDs, sentence_delimiter):
    """Extract articles from 2008-2010 updated/guided summary tasks 

    dataset_path: str, path to the parent directory of all docsets.
       Under the path, there are many folders, like D1022D, D1042H
        
    setIDs: list of str, consisting of "A" or "B"
        Most TAC summarization tasks uses two sets,
        each set consisting of 10 news articles

    sentence_delimiter: str, e.g.,"\n\n"

    return:
        dict, keys as document set (10 articles), values as list of 10 strings

    File structure: TAC20{08,09,10}_Summarization_Documents{.tgz}
                    |___> GuidedSumm{}_test_docs_files{.tar.gz}  (==dataset_path)
                          |___> D1001A
                                |___> D1001A-A (docset name is D1001-A)
                                      |___> 10 HTML files
                                |___> D1001A-B  (docset name is D1001-B, where the A is the NIST staff who picked this news)
                                      |___> 10 HTML files
                          |___> D1002A
                          ...
                          |___> D1046H 

    Todo: TAC2011 articles are released as indexes in Gigiword datasets. So a different function is needed. 

    """
    articles = {} 
    for docset in os.listdir(dataset_path):
        for set_suffix in setIDs:
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

def get_summaries(dataset_path, setIDs, sentence_delimiter, summary_types):
    """Extract summaries from 2008-2011 updated/guided summary tasks 

    dataset_path: str, path to a ROUGE (instead of peers or BE) directory, 
                 under which there are two folders: peers and models. 

    summary_types: list of strs, subsets of ["peers", "models"]

    setIDs: list of str, consisting of "A" or "B"
        Most TAC summarization tasks uses two sets,
        each set consisting of 10 news articles

    sentence_delimiter: str, e.g., " ---- " or "\n\n"

    return:
        dict, keys as document set (str, e.g., "D1001-B"), 
              values as dict, whose
                              keys as summarizer ID (str, e.g., "E" ), 
                              values as a list of strs, (summaries from 4 humans and 43 summarizers)
                              Each summary is a "\n"-separated string. 
 
    File structure: GuidedSumm20{08,09,10,11}_eval{.tgz}
                    |___> manual
                    |___> ROUGE  (==dataset_path)
                          |___> peers (summaries by machine summarizers)
                                |___> D1001-A.M.100.A.1 (leadword)
                                |___> D1001-A.M.100.A.2 (MEAD)
                                ...
                                |___> D1001-A.M.100.A.43 (3 to 43 are TAC participating summarizers)
                          |___> models (summaries by humans, 4 NIST staffers out of 8 total, A-H)
                                |___> D1001-A.M.100.A.{A-H} 
                    |___> BE    

    TODO: 
    Forrest: Please check whether it is one sentence per line. 

   """
    summaries = {} 
    for summary_type in summary_types:
        for summary_file in glob.glob(os.path.join(dataset_path,summary_type, "*")):
                    print (summary_file)
                    [docset_name, _, _, _, summarizer] = summary_file.split(".")
                    if docset_name not in summaries:
                        summaries[docset_name] = {}
                    with open(os.path.join(dataset_path, summary_type, summary_file), 'r') as f:
                        summary = f.read()
                        print (summary)
                    summaries[docset_name].setdefault(summarizer, []).append(summary)
    return summaries 
    

if __name__ == "__main__":
    article_set_path = "/mnt/insecure/data/TAC/TAC2010/TAC2010_Summarization_Documents/GuidedSumm10_test_docs_files/"
    summary_set_path = "/mnt/insecure/data/TAC/TAC2010/GuidedSumm2010_eval/ROUGE"

    setID = ["A"]
    sentence_delimiter = "  "
    summary_types = ["peers", "models"]

#    articles = get_articles(article_set_path, ["A"], sentence_delimiter)
#    print (get_statistics(articles))

    summaries = get_summaries(summary_set_path, ["A"], sentence_delimiter, summary_types)
