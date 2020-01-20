import json

if __name__ == "__main__":
    tac_file = "TAC2010_all.json"
    test_file = "TAC2010_test.tsv"

    tac = None
    with open(tac_file, "r", encoding="utf-8") as f:
        tac = json.load(f)

    with open(test_file, "w", encoding="utf-8") as f:
        ct = 0
        for doc in tac.keys():
            for article in tac[doc]['articles']:
                article = " ".join(article)
                article = article.replace("\n", " ")
                article = article.replace("\t", " ")

                if len(article) == 0:
                    article = "." 

                for summarizer in tac[doc]['summary'].keys():
                    summary = " ".join(tac[doc]['summary'][summarizer]['sentences'][0])
                    summary = summary.replace("\n", " ")
                    summary = summary.replace("\t", " ")
                    
                    scores = tac[doc]['summary'][summarizer]['scores']
                    scores = " ".join([str(score) for score in scores])

                    line = "\t".join([article, summary, scores]) + '\n'
                    f.write(line)

                    ct += 1
                    if ct % 1000 == 0:
                        print(ct)
        
        print(ct)
                    
                