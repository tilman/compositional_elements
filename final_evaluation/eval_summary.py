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
#EVAL_RESULTS_FILE = COMPOELEM_ROOT+"/final_evaluation/evaluation_log_grid_tune.pkl"
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
    "lexsort_hr_cr":"sortHrCr",
    "lexsort_cr_hr":"sortCrHr",
    "lexsort_md_hr":"sortHrMd",
    "sort_hr":"sortHr",
    "sort_cr":"sortCr",
    "sort_nccr1":"nccr1",
    "sort_nccr2":"nccr2",
    "sort_nccr3":"nccr3",
    "lexsort_ncosBuckets1_cr":"ncosB1Cr",
    "lexsort_ncosBuckets2_cr":"ncosB2Cr",
    "lexsort_ncosBuckets3_cr":"ncosB3Cr",
    "lexsort_fmr_cr":"lFmrCr",
    "lexsort_fmr_hr":"lFmrHr",
    "lexsort_cr_fmr":"lCrFmr",
    "lexsort_hr_fmr":"lHrFmr",
    "sort_fmrcr1":"sFmrcr1",
    "sort_fmrcr2":"sFmrcr2",
    "compare_pose_lines_2":"cmp2",
    "compare_pose_lines_3":"cmp3",
    "compare_siftFLANN2":"flann2",
    "compare_siftBFMatcher1":"bfm1",
    "compare_siftBFMatcher2":"bfm2",
    "compare_orbBFMatcher1":"bfm1",
    "compare_orbBFMatcher2":"bfm2",
    "compare_briefBFMatcher1":"bfm1",
    "compare_briefBFMatcher2":"bfm2",
    "compare_combinedSetupB":"cB",
    "compare_combinedSetupA":"cA",
    "compare_dist_min":"dist_min",
    "compare_dist_bipart":"dist_bipart",
}

def get_short_eval_name(log_entry):
    if "featuremap_key" in log_entry:
        return "{}|{}|{}".format(
            aliasNames[log_entry["datastore_name"]], 
            aliasNames[log_entry["featuremap_key"]], 
            aliasNames[log_entry["compare_method"]],
        )
    elif "setup" in log_entry:
        if "correction_angle" in log_entry:
          return "{}|{}|{}|{}|{}|ca{},co{},cs{}|th{}".format(
              aliasNames[log_entry["setup"]], 
              aliasNames[log_entry["datastore_name"]], 
              aliasNames[log_entry["norm_method"]], 
              aliasNames[log_entry["compare_method"]],
              aliasNames[log_entry["sort_method"]],
              log_entry["correction_angle"],
              log_entry["cone_opening_angle"],
              log_entry["cone_scale_factor"],
              " 75" if log_entry["filter_threshold"] == 75 else log_entry["filter_threshold"],
          )
        else:
          return "{}|{}|{}|{}|{}|ca{},co{},cs{}|th{}".format(
              aliasNames[log_entry["setup"]], 
              aliasNames[log_entry["datastore_name"]], 
              aliasNames[log_entry["norm_method"]], 
              aliasNames[log_entry["compare_method"]],
              aliasNames[log_entry["sort_method"]],
              20,
              80,
              10,
              " 75" if log_entry["filter_threshold"] == 75 else log_entry["filter_threshold"],
          )
    elif "combinedSetup" in log_entry:
        return "{}|{};|A|ceb|normGlAC|th150;img_vggBn".format(
            aliasNames[log_entry["combinedSetup"]],
            aliasNames[log_entry["sort_method"]],
        )
    elif "sift" in log_entry:
        return "sift|{}".format(
            aliasNames[log_entry["compare_method"]],
        )
    elif "orb" in log_entry:
        return "orb|{}".format(
            aliasNames[log_entry["compare_method"]],
        )
    elif "brief" in log_entry:
        return "brief|{}".format(
            aliasNames[log_entry["compare_method"]],
        )
    elif "linkingArt" in log_entry:
        return "linkingArt|{}".format(
            aliasNames[log_entry["compare_method"]],
        )
    else:
        return log_entry["experiment_id"]


# log = evaluation_log
new_log_entries = list(filter(lambda log_entry: log_entry["new"], evaluation_log))
# new_log_entries = list(filter(lambda log_entry: log_entry["new"], evaluation_log))
log = new_log_entries
display_metrics = ["p@1","p@5","p@10","p@50","r@1","r@5","r@10","r@50"]
a = pd.DataFrame([[get_short_eval_name(le), le['datetime'].strftime("%d.%m.%y %H:%M"), *le["eval_dataframe"].loc["total (mean)", display_metrics]] for le in log], columns=["short name", "datetime", *display_metrics])
pd.set_option('display.max_rows', None)
#print(a[-20:len(a)])
print(a)
#print(a.iloc[[1,23,36,49]]) # highscores for each method categorie
    
