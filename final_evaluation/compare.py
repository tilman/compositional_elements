import os
import numpy as np
import pickle
import shutil
import datetime

from tqdm.std import tqdm

from . import compare_deepfeatures
from . import compare_compoelem

#dataset_cleaned_extended_balanced = ceb_dataset -> combination of clean_data (all with _art classes nativity and virgin) dataset and files from prathmesn & ronak from 18.03.

DATASTORE_NAME = "combined_datastore_ceb_dataset"
DATASTORE_FILE = "/home/zi14teho/compositional_elements/final_evaluation/"+DATASTORE_NAME+".pkl" 
EVAL_RESULTS_FILE = "/home/zi14teho/compositional_elements/final_evaluation/evaluation_log.pkl"
#DATASET_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dataset_cleaned_extended_balanced"

datastore = pickle.load(open(DATASTORE_FILE, "rb"))
#dataset = np.array([[(className, img) for img in os.listdir( DATASET_ROOT+'/'+className)] for className in os.listdir( DATASET_ROOT )]).reshape(-1,2)
#classes, counts = np.unique(dataset[:,0], return_counts=True)
#print("total count:",len(dataset)," per class counts",list(zip(classes, counts)))

try:
    evaluation_log = pickle.load(open(EVAL_RESULTS_FILE, "rb"))
    for log_entry in evaluation_log:
        # log_entry["new"] = False
        pass
    shutil.copyfile(EVAL_RESULTS_FILE, EVAL_RESULTS_FILE+"_"+str(datetime.date.today())+"_backup")
except FileNotFoundError as e:
    evaluation_log = []
# [evaluation_log.append(experiment) for experiment in compare_deepfeatures.eval_all_combinations(datastore, DATASTORE_NAME, "imageNet_vgg19_bn_features")] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_deepfeatures.eval_all_combinations(datastore, DATASTORE_NAME, "places365_resnet50_feature_noFC")] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
[evaluation_log.append(experiment) for experiment in compare_compoelem.eval_all_combinations(datastore, DATASTORE_NAME)] 
pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))

def get_new_evaluation_log():
    evaluation_log = pickle.load(open(EVAL_RESULTS_FILE, "rb"))
    new_log_entries = list(filter(lambda log_entry: log_entry["new"], evaluation_log))
    return new_log_entries
print("new_log_entries: {}, evaluation_log_size:{}".format(len(get_new_evaluation_log()), len(evaluation_log)))
