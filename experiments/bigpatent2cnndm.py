import json
import gzip
import os
import sys
import hashlib
import argparse

if __name__ == "__main__":

    cpc_code = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'y']
    split_type = 'train'
    input_path = 'F:\\Dataset\\bigPatentData'
    output_path = "F:\\Dataset\\bigPatentData\\converted"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cnt = 0
    for code in cpc_code:
        file_names = os.listdir(os.path.join(input_path,split_type,code))
        for gz in file_names:
            with gzip.open(os.path.join(input_path,split_type,code,gz),'r') as fin:
                for row in fin:
                    md5 = hashlib.md5()
                    md5.update(row)
                    filename = md5.hexdigest()+".story"
                    filepath = os.path.join(output_path, filename)

                    obj = json.loads(row)
                    with open(filepath, "w", encoding="utf-8") as writer:
                        writer.write(obj["description"] + "\n\n@highlight\n\n" + obj["abstract"])
                    
                    cnt += 1
                    if cnt % 1000 == 0:
                        print("Converted %d" % cnt)
    
    print("Total %d" % cnt)

                    

