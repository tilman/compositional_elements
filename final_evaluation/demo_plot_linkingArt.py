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
query_data = data[0]
compare_results = []
for target_data in data:
    if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
        continue
    compare_results.append((compare_dist_min(query_data["compoelem"]["humans"], target_data["compoelem"]["humans"]), target_data)) # we will use the same precomputed poses as we already computed for compoelem method
compare_results = np.array(compare_results)
sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])][::-1]
query_label = query_data["className"]
res_labels = list(map(lambda x: x["className"], sorted_compare_results[:,-1]))
metrics = eval_utils.score_retrievals(query_label, res_labels)
print(metrics)

res_img = cv2.imread(DATASET_ROOT+'/'+query_data["className"]+'/'+query_data[["imgName"]])
res_img = visualize.pose_lines(res_data["pose_lines"], res_img) # type: ignore
res_img = visualize.global_action_lines(res_data["global_action_lines"], res_img) # type: ignore



fig, axis = plt.subplots(2, 6)

query_img = cv2.imread(DATASET_ROOT+'/'+query_data["className"]+'/'+query_data[["imgName"]])
query_img = visualize.pose_lines(res_data["pose_lines"], query_img) # type: ignore
query_img = visualize.global_action_lines(res_data["global_action_lines"], res_img) # type: ignore

query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
ax = axis[0][0] # type: ignore
ax.imshow(query_img_rgb)
ax.axis('off')
ax.set_title('query')

for res_idx, res in enumerate(sorted_compare_results[0:5]): #type: ignore
    #res_score, res_key, res_data = res # type: ignore => numpy unpack typing not fully supported
    combined_ratio, hit_ratio, mean_hit_dist, res_key, res_data = res # type: ignore => numpy unpack typing not fully supported
    #res_file_path = red_data["row"] # strip version number from key so we get filepath
    #res_img = cv2.imread(res_data["row"])
    res_img = download_img(res_data["row"])
    res_img = converter.resize(res_img)
    res_img = visualize.pose_lines(res_data["pose_lines"], res_img) # type: ignore
    res_img = visualize.global_action_lines(res_data["global_action_lines"], res_img) # type: ignore
    # visualize.safe("/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem/playground/res_img_"+str(res_idx+1)+".jpg", res_img)
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    # matplot:
    ax = axis[0][res_idx+1] # type: ignore
    ax.imshow(res_img_rgb)
    ax.axis('off')
    print("res_key", res_idx, res_data["row"]["image"])
    ax.set_title("cr:{:.3f}\nhr:{:.3f}\nhd:{:.3f}\n{}".format(combined_ratio, hit_ratio, mean_hit_dist, res_data["row"]["class"]))