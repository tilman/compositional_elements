# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import datetime
import torch
from tqdm import tqdm
from . import eval_utils
from compoelem.detect.openpose.lib.utils.common import BodyPart, Human, CocoPart
import copyreg
import pickle
from .compare_linkingArt  import compare_dist_min
from compoelem.detect.openpose.lib.utils.common import draw_humans


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

COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
DATASTORE_NAME = "combined_datastore_ceb_dataset"
DATASTORE_FILE = COMPOELEM_ROOT+"/final_evaluation/"+DATASTORE_NAME+".pkl"
DATASET_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dataset_cleaned_extended_balanced"

datastore = pickle.load(open(DATASTORE_FILE, "rb"))

data = list(datastore.values())
query_data = data[400]
compare_results = []
for target_data in data:
    if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
        continue
    distance, min_combination = compare_dist_min(query_data["compoelem"]["humans"], target_data["compoelem"]["humans"]) #type: ignore
    compare_results.append((distance, min_combination, target_data)) # we will use the same precomputed poses as we already computed for compoelem method
compare_results = np.array(compare_results)
sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])]
query_label = query_data["className"]
res_labels = list(map(lambda x: x["className"], sorted_compare_results[:,-1]))
metrics = eval_utils.score_retrievals(query_label, res_labels)
print(metrics)

#fig, axis = plt.subplots(1, 6)

RES_PLOT_SIZE = 3

fig, axis = plt.subplots(RES_PLOT_SIZE, 2)
for idx, res in enumerate(sorted_compare_results[0:RES_PLOT_SIZE]): #type: ignore
    score, combination, res_data = res # type: ignore => numpy unpack typing not fully supported
    idx_r, idx_s = combination
    query_img = cv2.imread(DATASET_ROOT+'/'+query_data["className"]+'/'+query_data["imgName"])
    res_img = draw_humans(query_img, query_data["compoelem"]["humans"][idx_r:idx_r+1])
    query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    res_img = cv2.imread(DATASET_ROOT+'/'+res_data["className"]+'/'+res_data["imgName"])
    res_img = draw_humans(res_img, res_data["compoelem"]["humans"][idx_s:idx_s+1])
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    ax = axis[idx]
    ax[0].imshow(query_img_rgb)
    ax[0].axis('off')
    ax[0].set_title('{}'.format(query_data["className"]))
    ax[1].imshow(res_img_rgb)
    ax[1].axis('off')
    print("res_key", idx, res_data["imgName"])
    ax[1].set_title("score:{:.3f} {}".format(score, res_data["className"]))

#plt.tight_layout()
# plt.subplots_adjust(top=0.85) # Make space for title
plt.show()