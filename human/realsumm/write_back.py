import pickle
import copy
import re
import os

def main():    
    sd_abs_path = "abs.pkl"
    sd_ext_path = "ext.pkl"
    sd_abs = pickle.load(open(sd_abs_path, "rb"))
    sd_ext = pickle.load(open(sd_ext_path, "rb"))
    sd = copy.deepcopy(sd_abs)
    abs_systems = sd_abs[1]['system_summaries'].keys()
    ext_systems = sd_ext[1]['system_summaries'].keys()
    for doc_id in sd:
        isd_sota_ext = sd_ext[doc_id]
        isd_sota_ext['system_summaries']['bart_out_ext.txt'] = isd_sota_ext['system_summaries']['bart_out.txt']
        sd[doc_id]['system_summaries'].update(isd_sota_ext['system_summaries'])
        del isd_sota_ext['system_summaries']['bart_out.txt']

    print(list(abs_systems))
    print(list(ext_systems))
    print("bart_out_ext.txt" in ext_systems)

    pred_path = 'predictions'
    preds = os.listdir(pred_path)
    preds = [tsvfile for tsvfile in preds if tsvfile.endswith('.tsv')]
    for tsvfile in preds:
        our_metric = "ours_" + tsvfile.split('.')[0] if not tsvfile.startswith("metric") else tsvfile.split('.')[0]
        with open(os.path.join(pred_path, tsvfile), "r", encoding="utf-8") as f:
            for doc_id in sd:
                for sys_name, system in sd[doc_id]["system_summaries"].items():
                    score = float(f.readline())
                    
                    if sys_name in abs_systems:
                        sd_abs[doc_id]["system_summaries"][sys_name]["scores"][our_metric] = score
                    elif sys_name in ext_systems:
                        sd_ext[doc_id]["system_summaries"][sys_name]["scores"][our_metric] = score
                    else:
                        print("???")
                        assert(False)
    
    new_sd_abs_path = "../scores_dicts/abs_ours.pkl"
    new_sd_ext_path = "../scores_dicts/ext_ours.pkl"
    pickle.dump(sd_abs, open(new_sd_abs_path, "wb"))
    pickle.dump(sd_ext, open(new_sd_ext_path, "wb"))

if __name__ == '__main__':
    main()