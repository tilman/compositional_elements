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
from .compare_linkingArt  import compare_dist_min, compare_dist_bipart, sort_asc, robust_verify
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
#query_data = data[0] #400
#query_data = datastore["annunciation_page_8_item_11_3annunc.jpg"]
query_data = datastore["adoration_page_11_item_15_adormag1.jpg"]
print(query_data["className"])
print("a", query_data["className"])
compare_results = []
for target_data in data:
    if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
        continue
    distance, combination = compare_dist_min(query_data["compoelem"]["humans"], target_data["compoelem"]["humans"])
    #[28 0.48810905977318275 list([(4, 6), (5, 4), (6, 7)])
    compare_results.append((distance, combination, target_data)) # we will use the same precomputed poses as we already computed for compoelem method
compare_results = np.array(compare_results)
sorted_compare_results = sort_asc(compare_results)

l = 50
# TODO: cut sorted_compare_results into first segment [0:l] and remaining segment [l:-1]
# for first segment perform RANSAC (robust verification) and set matched/unmatched
        # => TODO: pose pairs whose distance exceeds 0.1 are discarded
# for last segment set all to unmatched
# then perform lexsort, Sort in a manner that all matched ones comes to the front and unmatched to the back. Second criteria is than distance from above
# compare_results = [(0, *r) for r in sorted_compare_results[l:-1]] #TODO check if padding with 0 is really what we want. Since idx0 stand for max(total_consistent) in the robust_verify result i guess so
# for target in sorted_compare_results[0:l]:
#     target_data = target[-1]
#     if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
#         continue
#     matched = robust_verify(query_data["compoelem"]["humans"], target_data["compoelem"]["humans"], neck_norm=True)
#     compare_results.append((matched, *target))
# compare_results = np.array(compare_results)
# sorted_compare_results = compare_results[np.lexsort((compare_results[:,1], -compare_results[:,0]))] # first level of sorting is 0 (verification), and then 1 (distance)

query_label = query_data["className"]
res_labels = list(map(lambda x: x["className"], sorted_compare_results[:,-1]))
metrics = eval_utils.score_retrievals(query_label, res_labels)
#print(metrics)

#fig, axis = plt.subplots(1, 6)

RES_PLOT_SIZE = 5

fig, axis = plt.subplots(1, RES_PLOT_SIZE+1)
query_img = cv2.imread(DATASET_ROOT+'/'+query_data["className"]+'/'+query_data["imgName"])
query_img = draw_humans(query_img, query_data["compoelem"]["humans"])
query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
axis[0].imshow(query_img_rgb)
axis[0].axis('off')
axis[0].set_title('query\n{}'.format(query_label))

for idx, res in enumerate(sorted_compare_results[0:RES_PLOT_SIZE]): #type: ignore
    print("idx",idx)
    distance, combination, res_data = res # type: ignore => numpy unpack typing not fully supported
    print("dist",distance,combination)
    res_img = cv2.imread(DATASET_ROOT+'/'+res_data["className"]+'/'+res_data["imgName"])

    # draw all poses:
    #query_img_all = draw_humans(np.array(query_img), query_data["compoelem"]["humans"])
    # res_img_all = draw_humans(np.array(res_img), res_data["compoelem"]["humans"])

    # draw all matched poses:
    # for idx_r, idx_s in combinations:
    #query_img = draw_humans(query_img, [query_data["compoelem"]["humans"][combination[0]]])
    res_img = draw_humans(res_img, [res_data["compoelem"]["humans"][combination[1]]])

    # query_img_all_rgb = cv2.cvtColor(query_img_all, cv2.COLOR_BGR2RGB)
    # query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    # res_img_all_rgb = cv2.cvtColor(res_img_all, cv2.COLOR_BGR2RGB)
    ax = axis[idx+1]
    ax.imshow(res_img_rgb)
    ax.axis('off')
    ax.set_title('retrieval {}\n{}'.format(idx+1,res_data["className"]))

    # ax[0].imshow(query_img_all_rgb)
    # ax[0].axis('off')
    # ax[0].set_title('all query poses')
    # ax[1].imshow(query_img_rgb)
    # ax[1].axis('off')
    # ax[1].set_title('matched poses {}'.format(query_data["className"]))
    # ax[2].imshow(res_img_rgb)
    # ax[2].axis('off')
    # ax[2].set_title("matched poses\nmatched distance{:.3f} {}".format(distance, res_data["className"]))
    # ax[3].imshow(res_img_all_rgb)
    # ax[3].axis('off')
    # ax[3].set_title("all target poses")
    # print("res_key", idx, res_data["imgName"])

#plt.tight_layout()
# plt.subplots_adjust(top=0.85) # Make space for title
plt.show()