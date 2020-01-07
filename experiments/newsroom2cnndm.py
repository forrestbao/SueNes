import json, gzip
import os
import hashlib

if __name__ == '__main__':
    NEWSROOM_DIR = "F:\\Dataset\\newsroom\\release"
    path = os.path.join(NEWSROOM_DIR, "train.jsonl.gz")
    OUTPUT_DIR = "F:\\Dataset\\newsroom\\converted"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with gzip.open(path) as f:
        cnt = 0
        for ln in f:
            md5 = hashlib.md5()
            md5.update(ln)
            filename = md5.hexdigest()+".story"
            filepath = os.path.join(OUTPUT_DIR, filename)

            obj = json.loads(ln) 
            with open(filepath, "w", encoding="utf-8") as writer:
                writer.write(obj["text"] + "\n\n@highlight\n\n" + obj["summary"])
            
            cnt += 1
            if cnt % 1000 == 0:
                print("Converted %d" % cnt)
        
        print("Total %d" % cnt)
    