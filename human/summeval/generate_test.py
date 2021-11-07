import re
import json

def main():
    with open("model_annotations.aligned.paired.scored.jsonl", "r", encoding="utf-8") as fd:
        dataset = [json.loads(line) for line in fd]

    with open("summeval_100.tsv", "w", encoding="utf-8") as f:
        for example in dataset:
            # print(sd[doc_id]["doc_id"], doc_id)
            doc_src = example["text"]
            doc_src = doc_src.replace("\t", " ")
            doc_src = doc_src.strip()
            sys_sum = example["decoded"]
            sys_sum = sys_sum.replace("\t", " ")
            sys_sum = sys_sum.strip()
            
            linestr = "\t".join([doc_src, sys_sum])
            linestr = re.sub(" +", " ", linestr)
            f.write(linestr + "\n")

if __name__ == '__main__':
    main()