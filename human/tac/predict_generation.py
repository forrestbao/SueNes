import json
from tac import get_rouge

if __name__ == "__main__":
    tac_file = "TAC2010_all.json"
    test_file = "TAC2010_test.tsv"
    human = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}

    rouge_score_path = "F:\\Dataset\\TAC2010/TAC2010/GuidedSumm2010_eval/ROUGE/rouge_A.m.out"
    rouge_score = get_rouge(rouge_score_path, None)
    print(rouge_score.keys())

    tac = None
    with open(tac_file, "r", encoding="utf-8") as f:
        tac = json.load(f)

    with open(test_file, "w", encoding="utf-8") as f:
        ct = 0
        
        for doc in tac.keys():
            for summarizer in tac[doc]['summary'].keys():
                summary = " ".join(tac[doc]['summary'][summarizer]['sentences'][0])
                summary = summary.replace("\n", " ")
                summary = summary.replace("\t", " ")
                if len(summary) == 0:
                    summary = "."
                
                scores = tac[doc]['summary'][summarizer]['scores']
                scores = "\t".join([str(score) for score in scores])
                rouge = ""
                if summarizer not in human:
                    rouge = "\t".join([str(score) for score in rouge_score[(doc, summarizer)][3:6]])

                for article in tac[doc]['articles']:
                    article = " ".join(article)
                    article = article.replace("\n", " ")
                    article = article.replace("\t", " ")

                    if len(article) == 0:
                        article = "." 
                    
                    line = "\t".join([article, summary, scores, "0" if summarizer in human else "1", rouge]) + '\n'
                    f.write(line)

                    ct += 1
                    if ct % 1000 == 0:
                        print(ct)
        
        print(ct)
                    
                