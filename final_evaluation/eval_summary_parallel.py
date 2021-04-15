import pickle
import os
import numpy as np
import pandas as pd

if os.uname().nodename == 'MBP-von-Tilman':
    COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
elif os.uname().nodename == 'lme117':
    COMPOELEM_ROOT = "/home/zi14teho/compositional_elements"
else:
    COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
#EVAL_RESULTS_FILE = COMPOELEM_ROOT+"/final_evaluation/evaluation_log_grid_tune.pkl"
EVAL_RESULTS_FILE = COMPOELEM_ROOT+"/final_evaluation/evaluation_log.pkl"

evaluation_log = []
#[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth150_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth150_noFb", "rb"))]
#[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth200_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth200_noFb", "rb"))]
#[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth250_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth250_noFb", "rb"))]
#[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth300_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth300_noFb", "rb"))]
# new second batch
for le in pickle.load(open(".tmpEvalLog_fth150_onlyPoseFb","rb")):
    le["prefix"] = "pl"
    evaluation_log.append(le)
for le in pickle.load(open(".tmpEvalLog_fth150_onlyPoseFb_normGacFallback","rb")):
    le["prefix"] = "pl,ar"
    evaluation_log.append(le)
for le in pickle.load(open(".tmpEvalLog_fth200_onlyPoseFb","rb")):
    le["prefix"] = "pl"
    evaluation_log.append(le)
for le in pickle.load(open(".tmpEvalLog_fth200_onlyPoseFb_normGacFallback","rb")):
    le["prefix"] = "pl,ar"
    evaluation_log.append(le)
for le in pickle.load(open(".tmpEvalLog_fth250_onlyPoseFb","rb")):
    le["prefix"] = "pl"
    evaluation_log.append(le)
for le in pickle.load(open(".tmpEvalLog_fth250_onlyPoseFb_normGacFallback","rb")):
    le["prefix"] = "pl,ar"
    evaluation_log.append(le)
for le in pickle.load(open(".tmpEvalLog_fth300_onlyPoseFb","rb")):
    le["prefix"] = "pl"
    evaluation_log.append(le)
for le in pickle.load(open(".tmpEvalLog_fth300_onlyPoseFb_normGacFallback","rb")):
    le["prefix"] = "pl,ar"
    evaluation_log.append(le)

# evaluation_log = pickle.load(open(EVAL_RESULTS_FILE, "rb"))
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
          #return "${};\\rho{},\\omega={},\\sigma={},\\eta={};\\beta{}$".format(
          return "{} & {} & {} & {} & {} & {}".format(
          
            #   aliasNames[log_entry["setup"]], 
              log_entry["prefix"] if "prefix" in log_entry else ("pl,bi" if log_entry["with_fallback"] else "x"),
            #   aliasNames[log_entry["datastore_name"]], 
            #   aliasNames[log_entry["norm_method"]], 
            #   aliasNames[log_entry["compare_method"]],
            #   aliasNames[log_entry["sort_method"]],
              log_entry["correction_angle"],
              log_entry["cone_opening_angle"],
              log_entry["cone_scale_factor"],
              log_entry["cone_base_scale_factor"],
              " 75" if log_entry["filter_threshold"] == 75 else log_entry["filter_threshold"],
        #   return "{}|{}|{}|{}|{}|{}|ca{},co{},cs{}|th{}".format(
        #       aliasNames[log_entry["setup"]], 
        #       "wFb" if log_entry["with_fallback"] else "nFb",
        #       aliasNames[log_entry["datastore_name"]], 
        #       aliasNames[log_entry["norm_method"]], 
        #       aliasNames[log_entry["compare_method"]],
        #       aliasNames[log_entry["sort_method"]],
        #       log_entry["correction_angle"],
        #       log_entry["cone_opening_angle"],
        #       log_entry["cone_scale_factor"],
        #       log_entry["cone_base_scale_factor"],
        #       " 75" if log_entry["filter_threshold"] == 75 else log_entry["filter_threshold"],
          )
        else:
          return "{}|{}|{}|{}|{}|ca{},co{},cs{},csb{}|th{}".format(
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
display_metrics = ["p@1","p@2","p@3","p@5","p@10","p@50","r@1","r@5","r@10","r@50"]
a = pd.DataFrame([
    [
        get_short_eval_name(le), 
        le['datetime'].strftime("%d.%m.%y %H:%M"), 
        *le["eval_dataframe"].loc["total (mean)", display_metrics],
        np.mean(le["eval_dataframe"].loc["total (mean)", ["p@1","p@2","p@3","p@5","p@10"]]),
        np.mean(le["eval_dataframe"].loc["total (mean)", ["r@1","r@2","r@3","r@5","r@10"]]),
    ] for le in log], columns=["short name", "datetime", *display_metrics, "p@1-p@10 mean", "r@1-r@10 mean"]).sort_values("p@1")
# pd.set_option('display.max_rows', None)
#print(a[-20:len(a)])
print(a[-10:len(a)])

for r in a.iloc[-10:len(a)][["short name", "p@1", "r@1", "p@1-p@10 mean", "r@1-r@10 mean"]].to_numpy()[::-1]:
    name, p1, r1, p1_10_mean, r1_10_mean = r
    p1 = round(p1*100,3)
    r1 = round(r1*100,4)
    p1_10_mean = round(p1_10_mean*100,3)
    r1_10_mean = round(r1_10_mean*100,4)
    line = "{}        &    {}\\%    &   {}\\%    &  {}\\%   &   {}\\% \\\\".format(name, p1, r1, p1_10_mean, r1_10_mean)
    print(line)


#print(a.iloc[[1,23,36,49]]) # highscores for each method categorie