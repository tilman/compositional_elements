import pickle
import os
import numpy as np
import pandas as pd

osuname = os.uname().nodename
print("osuname", osuname)
if osuname == 'MBP-von-Tilman' or osuname == 'MacBook-Pro-von-Tilman.local':
    COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
elif osuname == 'lme117':
    COMPOELEM_ROOT = "/home/zi14teho/compositional_elements"
else:
    COMPOELEM_ROOT = os.getenv('COMPOELEM_ROOT')
DATASTORE_NAME = "combined_datastore_ceb_dataset"
DATASTORE_FILE = COMPOELEM_ROOT+"/final_evaluation/"+DATASTORE_NAME+".pkl"
EVAL_RESULTS_FILE_DIR = COMPOELEM_ROOT+"/final_evaluation/final2pkl/"


evaluation_log = [pickle.load(open(EVAL_RESULTS_FILE_DIR+"/"+logfile, "rb")) for logfile in os.listdir( EVAL_RESULTS_FILE_DIR )]
print(len(evaluation_log))
# new_log_entries = list(filter(lambda log_entry: log_entry["new"], evaluation_log))
# log = new_log_entries


display_metrics = ["p@1","p@2","p@3","p@5","p@10","p@20","p@30","p@50","p@rel","mAP","r@1","r@2","r@3","r@5","r@10","r@20","r@30","r@50","r@rel","mAR"]
#display_metrics = ["p@1"]
#display_metrics = ["p@1"]#,"p@2","p@3"]
#a = pd.DataFrame([[ le['experiment_name'], le['filename'][24:-4], le['datetime'].strftime("%d.%m.%y %H:%M"), *le["eval_dataframe"].loc["total (mean)", display_metrics] ] for le in evaluation_log], columns=["experiment_name", "name", "date", *display_metrics])
#a = pd.DataFrame([[ le['experiment_name'], le['filename'][24:-4], le['datetime'].strftime("%d.%m.%y-%H"), *le["eval_dataframe"].loc["total (mean)", display_metrics] ] for le in evaluation_log], columns=["experiment_name", "name", "date", *display_metrics])


a = pd.DataFrame([
    [
        le,
        le['experiment_name'], 
        #le['filename'][24:-4],
        le['filename'][13:56],
        le['datetime'].strftime("%d.%m.%y %H:%M"), 
        *le["eval_dataframe"].loc["total (mean)", display_metrics],
        np.mean(le["eval_dataframe"].loc["total (mean)", ["p@1","p@2","p@3","p@5","p@10"]]),
        np.mean(le["eval_dataframe"].loc["total (mean)", ["r@1","r@2","r@3","r@5","r@10"]]),
    ] for le in evaluation_log], columns=["log_entry", "experiment_name", "name", "datetime", *display_metrics, "p@1-p@10 mean", "r@1-r@10 mean"]).sort_values("datetime")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
#a = a[a['name'] == "normGlac_cr_desc_ca20_co80_cs10_cbs0_th150_fbPlTrue_fbBisFalse_fbGaTrue"]
# print(a.sort_values("date"), len(a))
#print(a[-30:len(a)].sort_values("experiment_name")[["experiment_name", "name", "p@1"]])
print(a[-30:len(a)].sort_values("experiment_name")[["experiment_name", "name", "p@1", "p@2", "p@5", "p@10"]])

# for r in a.iloc[-10:len(a)][["name", "p@1", "r@1", "p@1-p@10 mean", "r@1-r@10 mean"]].to_numpy()[::-1]:
#    name, p1, r1, p1_10_mean, r1_10_mean = r
#    p1 = round(p1*100,2)
#    r1 = round(r1*100,4)
#    p1_10_mean = round(p1_10_mean*100,4)
#    r1_10_mean = round(r1_10_mean*100,4)
#    line = "{}        &    {}\\%    &   {}\\%    &  {}\\%   &   {}\\% \\\\".format(name, p1, r1, p1_10_mean, r1_10_mean)
#    print(line)