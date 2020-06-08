from os import listdir
from os.path import isfile, join
import re
import json


infolder = '/work/Data/cnndm-30k'

for folder in ['train', 'test']:
    for data in ['mutate/add', 'mutate/delete', 'mutate/replace', 'neg']:
        infile = '%s/%s-%s.tsv' % (infolder, re.sub('/', '-', data), folder)
        outfile = '%s/%s-%s2.tsv' % (infolder, re.sub('/', '-', data), folder)
        num = 0
        fw = open(outfile, 'w')
        for line in open(infile, 'r'):
            jdict = json.loads(line)
            #print(jdict[0].keys())
            #print(jdict[1].keys())
            num += 1
            num += 1
            article = re.sub('\t', ' ', jdict[0]['article'])
            summary = re.sub('\t', ' ', jdict[0]['summary'])
            article2 = re.sub('\t', ' ', jdict[1]['article'])
            summary2 = re.sub('\t', ' ', jdict[1]['summary'])
            fw.write('%s\t%s\t%s\n'%(article, summary, jdict[0]['label']))
            fw.write('%s\t%s\t%s\n'%(article2, summary2, jdict[1]['label']))
        print(num, 'lines in', outfile)
        fw.close()

