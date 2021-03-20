import pickle
import os
import numpy as np
import pandas as pd

if os.uname().nodename == 'MBP-von-Tilman':
    COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
elif os.uname().nodename == 'lme117':
    COMPOELEM_ROOT = "/home/zi14teho/compositional_elements"
else:
    COMPOELEM_ROOT = os.getenv('COMPOELEM_ROOT')
EVAL_RESULTS_FILE = COMPOELEM_ROOT+"/final_evaluation/evaluation_log.pkl"
evaluation_log = pickle.load(open(EVAL_RESULTS_FILE, "rb"))
aliasNames = {
    "eucl_dist_flatten":              "eucl",
    "negative_cosine_dist_flatten":   "ncos",
    "combined_datastore_ceb_dataset": "ceb",
    "imageNet_vgg19_bn_features":     "img_vggBn",
    "places365_resnet50_feature_noFC":"plc_res50",
    "compare_setupA":"A",
    "compare_setupB":"B",
    "minmax_norm_by_imgrect":"normRect",
    "minmax_norm_by_bbox":"normBBox",
    "norm_by_global_action":"normGlAC",
    "lexsort_hr_md":"sortHrMd",
    "lexsort_cr_hr":"sortCrHr",
}

def get_short_eval_name(log_entry):
    if "featuremap_key" in log_entry:
        return "{}|{}|{}".format(
            aliasNames[log_entry["datastore_name"]], 
            aliasNames[log_entry["featuremap_key"]], 
            aliasNames[log_entry["compare_method"]],
        )
    if "setup" in log_entry:
        return "{}|{}|{}|{}|th{}".format(
            aliasNames[log_entry["setup"]], 
            aliasNames[log_entry["datastore_name"]], 
            aliasNames[log_entry["norm_method"]], 
            #aliasNames[log_entry["compare_method"]],
            aliasNames[log_entry["sort_method"]],
            log_entry["config"]["compare"]["filter_threshold"],
        )

#new_log_entries = list(filter(lambda log_entry: log_entry["new"], evaluation_log))
# log = new_log_entries
log = evaluation_log
display_metrics = ["p@1","p@5","p@10","p@50","r@1","r@5","r@10","r@50"]
a = pd.DataFrame([[get_short_eval_name(le), le['datetime'].strftime("%d.%m.%y %H:%M"), *le["eval_dataframe"].loc["total (mean)", display_metrics]] for le in log], columns=["short name", "datetime", *display_metrics])
print(a)
    
