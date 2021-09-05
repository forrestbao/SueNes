# Convert newsroom's human annotation data to two-column TSV 
# Human annotation from https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv

import csv
import html

pairs = []
counter = 0 
with open('newsroom-human-eval.csv', ) as csvfile: 
    reader = csv.reader(csvfile, delimiter=",", quotechar="\"") 
    for row in reader: 
        if counter > 0:
            [_doc, _sum] = row[2:4]
            _doc = _doc.replace("</p><p>", "")
            _sum = _sum.replace("</p><p>", "")
            _doc=html.unescape(_doc) 
            _sum=html.unescape(_sum) 

            pairs.append([_doc, _sum])
        counter += 1
    
print (len(pairs))

with open('newsroom_60.tsv', 'w') as csvfile: 
    for pair in pairs:
        csvfile.write("\t".join(pair)+"\n")
# I did not use csv.writer because it inserts quotation marks in a way I cannot get 

# find newsroom_60.tsv -type d -exec cp /mnt/12T/data/NLP/anti-rogue/result_base_sent {} \;