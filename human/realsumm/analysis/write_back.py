import pickle
import utils
import copy
import re

def main():
    our_metric = "ours_bert_scientific_paper"    
    
    sd_abs_path = "../scores_dicts/abs.pkl"
    sd_ext_path = "../scores_dicts/ext.pkl"
    sd_abs = utils.get_pickle(sd_abs_path)
    sd_ext = utils.get_pickle(sd_ext_path)
    sd = copy.deepcopy(sd_abs)
    abs_systems = sd_abs[1]['system_summaries'].keys()
    ext_systems = sd_ext[1]['system_summaries'].keys()
    print(list(abs_systems))
    print(list(ext_systems))
    for doc_id in sd:
        isd_sota_ext = sd_ext[doc_id]
        isd_sota_ext['system_summaries']['bart_out_ext.txt'] = isd_sota_ext['system_summaries']['bart_out.txt']
        sd[doc_id]['system_summaries'].update(isd_sota_ext['system_summaries'])

    with open("test_results.tsv", "r", encoding="utf-8") as f:
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