from os import listdir
from os.path import isfile, join
import re


infolder = '/work/Data/cnndm-30k'

for folder in ['train', 'test']:
    for data in ['mutate/add', 'mutate/delete', 'mutate/replace', 'neg']:
        outfile = '%s/%s-%s.tsv' % (infolder, re.sub('/', '-', data), folder)
        fw = open(outfile, 'w')
        num = 0
        mypath = '%s/%s/%s/' % (infolder, data, folder)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for infile in onlyfiles:
            for line in open(mypath + infile, 'r'):
                if line.strip() == '':
                    continue
                fw.write(line.strip() + '\n')
                num += 1
        fw.close()
        print('total lines', num, outfile)

