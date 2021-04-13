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


#display_metrics = ["p@1","p@5","p@10","p@50","r@1","r@5","r@10","r@50"]
display_metrics = ["p@1","p@5","p@10","p@50"]
a = pd.DataFrame([[ le['experiment_name'], le['filename'][24:-4], le['datetime'].strftime("%d.%m.%y %H:%M"), *le["eval_dataframe"].loc["total (mean)", display_metrics] ] for le in evaluation_log], columns=["experiment_name", "name", "date", *display_metrics]).sort_values("experiment_name")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
print(a)