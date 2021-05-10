import os
import numpy as np
import pickle
import shutil
import datetime
import cv2
import copyreg

from tqdm.std import tqdm

from . import compare_deepfeatures
from . import compare_compoelem_new
from . import compare_combined_vgg19
from . import compare_combined_sift
from . import compare_linkingArt
from . import compare_sift
from . import compare_orb
from . import compare_brief



# fix cv2 keypoint pickling error
def _pickle_keypoint(keypoint): #  : cv2.KeyPoint
    return cv2.KeyPoint, (
        keypoint.pt[0],
        keypoint.pt[1],
        keypoint.size,
        keypoint.angle,
        keypoint.response,
        keypoint.octave,
        keypoint.class_id,
    )
# Apply the bundling to pickle
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)

#dataset_cleaned_extended_balanced = ceb_dataset -> combination of clean_data (all with _art classes nativity and virgin) dataset and files from prathmesn & ronak from 18.03.

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
EVAL_RESULTS_FILE = COMPOELEM_ROOT+"/final_evaluation/evaluation_log.pkl"

datastore = pickle.load(open(DATASTORE_FILE, "rb"))

try:
    evaluation_log = pickle.load(open(EVAL_RESULTS_FILE, "rb"))
    # for log_entry in evaluation_log:
    #     log_entry["new"] = False
    shutil.copyfile(EVAL_RESULTS_FILE, EVAL_RESULTS_FILE+"_"+str(datetime.date.today())+"_backup")
except FileNotFoundError as e:
    evaluation_log = []
# [evaluation_log.append(experiment) for experiment in compare_deepfeatures.eval_all_combinations(datastore, DATASTORE_NAME, "imageNet_vgg19_bn_features")] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_deepfeatures.eval_all_combinations(datastore, DATASTORE_NAME, "places365_resnet50_feature_noFC")] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_compoelem.eval_all_combinations(datastore, DATASTORE_NAME)] 

#fallback: yes, no
#filter_threshold: 150, 200, 250, 300
# [evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 150, True)]
# [evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 200, True)]
# [evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 250, True)]
# [evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 300, True)]
# [evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 150, False)]
[evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 200, False)]
# [evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 250, False)]
# [evaluation_log.append(experiment) for experiment in compare_compoelem_new.eval_all_combinations(datastore, DATASTORE_NAME, 300, False)]
# def eval_all_combinations(datastore, datastore_name, filter_threshold, with_fallback):
try:
    evaluation_log = pickle.load(open(EVAL_RESULTS_FILE, "rb"))
    pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
except Exception as e:
    print("open err",e)
# [evaluation_log.append(experiment) for experiment in compare_combined_vgg19.eval_all_combinations(datastore, DATASTORE_NAME)] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_sift.eval_all_combinations(datastore, DATASTORE_NAME)] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_orb.eval_all_combinations(datastore, DATASTORE_NAME)] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_brief.eval_all_combinations(datastore, DATASTORE_NAME)] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_combined_sift.eval_all_combinations(datastore, DATASTORE_NAME)] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))
# [evaluation_log.append(experiment) for experiment in compare_linkingArt.eval_all_combinations(datastore, DATASTORE_NAME)] 
# pickle.dump(evaluation_log, open(EVAL_RESULTS_FILE, "wb"))

def get_new_evaluation_log():
    evaluation_log = pickle.load(open(EVAL_RESULTS_FILE, "rb"))
    new_log_entries = list(filter(lambda log_entry: log_entry["new"], evaluation_log))
    return new_log_entries
print("new_log_entries: {}, evaluation_log_size:{}".format(len(get_new_evaluation_log()), len(evaluation_log)))
