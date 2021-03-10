# call this script with `python -m evaluation.evaluate_poselines_globalaction`
from matplotlib import pyplot as plt
from compoelem.types import PoseLine
import os
import requests
import shutil
import sys
from typing import Any, Sequence, Tuple, cast
import numpy as np
import pickle
import time
from compoelem.config import config

import cv2
from tqdm import tqdm
from compoelem.generate import global_action, pose_abstraction, pose_direction
from compoelem.visualize import visualize
from compoelem.detect import converter
from compoelem.compare.pose_line import compare_pose_lines_2, filter_pose_line_ga_result
from compoelem.compare.normalize import minmax_norm_by_imgrect, norm_by_global_action





def score_retrievals(label, retrievals):
    """
    Evaluating the current retrieval experiment
    Args:
    -----
    label: string
        label corresponding to the query
    retrivals: list
        list of strings containing the ranked labels corresponding to the retrievals
    tot_labels: integer
        number of images with the current label. We need this to compute recalls
    """
    # retrievals = retrievals[1:] # we do not account rank-0 since it's self-retrieval
    relevant_mask = np.array([1 if r==label else 0 for r in retrievals])
    num_relevant_retrievals = np.sum(relevant_mask)
    if(num_relevant_retrievals == 0):
        print(label)
        metrics = {
            "label": label,
            "p@1": -1,
            "p@5": -1,
            "p@10": -1,
            "p@rel": -1,
            "mAP": -1,
            "r@1": -1,
            "r@5": -1,
            "r@10": -1,
            "r@rel": -1,
            "mAR": -1
        }
        return metrics
    # computing precision based metrics
    precision_at_rank = np.cumsum(relevant_mask) / np.arange(1, len(relevant_mask) + 1)
    precision_at_1 = precision_at_rank[0]
    precision_at_5 = precision_at_rank[4]
    precision_at_10 = precision_at_rank[9]
    precision_at_rel = precision_at_rank[num_relevant_retrievals - 1]
    average_precision = np.sum(precision_at_rank * relevant_mask) / num_relevant_retrievals
    # computing recall based metrics
    recall_at_rank = np.cumsum(relevant_mask) / num_relevant_retrievals
    recall_at_1 = recall_at_rank[0]
    recall_at_5 = recall_at_rank[4]
    recall_at_10 = recall_at_rank[9]
    recall_at_rel = recall_at_rank[num_relevant_retrievals - 1]
    average_recall = np.sum(recall_at_rank * relevant_mask) / num_relevant_retrievals
    metrics = {
        "label": label,
        "p@1": precision_at_1,
        "p@5": precision_at_5,
        "p@10": precision_at_10,
        "p@rel": precision_at_rel,
        "mAP": average_precision,
        "r@1": recall_at_1,
        "r@5": recall_at_5,
        "r@10": recall_at_10,
        "r@rel": recall_at_rel,
        "mAR": average_recall
    }
    return metrics




# from compoelem.detect.openpose.lib.utils.common import draw_humans
DATASTORE_CACHE_DIR = ".cache/"

def download_img(row):
    #print(row)
    filename = DATASTORE_CACHE_DIR+row["image"]
    img = cv2.imread(filename)
    if img is not None:
        return img
    r = requests.get(row["url"], stream = True)
    if r.status_code == 200:
        r.raw.decode_content = True
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print('Image sucessfully Downloaded:')
    else:
        print('Image Couldn\'t be retreived')
    img = cv2.imread(filename)
    return img




#STORAGE_VERSION_KEY = "v1"
DATASTORE_FILE = os.path.join(os.path.dirname(__file__), "../rebalanced_datastore_icon_title_2_noRape.pkl")
datastore = pickle.load(open(DATASTORE_FILE, "rb"))
# key A
# query_image_key = 'v1/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/old_icc/icc_images_imdahl/flucht-nach-aegypten.jpg'
# # key B
# query_image_key = 'v1/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/old_icc/icc_images_imdahl/Giotto-Di-Bondone-Flight-into-Egypt-2-.jpeg'
# # key C
# query_image_key = 'v1/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dirk_27.11.2020_Sample_Adoration_Annunciation_Baptism/Adoration/A6_Adriaan_de_Weerdt_-_Adoration_of_the_Magi.jpg'
#key 2A
#query_image_key="v1/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/til_selection_icon_title_dataset_2/_art_g_ghirland_domenico_1.jpg"
query_image_key=list(datastore.keys())[0]

#TODO prathmesh idx 350 request
# res of idx 350
# res_key 0 _art_zgothic_mosaics_6sanmarc_9baptis1.jpg
# res_key 1 _art_l_loir_n_adorshep.jpg
# res_key 2 _art_m_master_straus_adormagi.jpg
# res_key 3 _art_v_veronese_08_collegio_5anticoy.jpg -> and r3 of r3: _art_s_sustris_lambert_baptism.jpg
# res_key 4 _art_g_greco_el_02_0207gred.jpg
# res_key 5 _art_m_montanez_shepherd.jpg
# res_key 6 _art_a_albani_1_annuncia.jpg
query_image_key="_art_a_albani_1_annuncia.jpg"
print("query_image_key", query_image_key)

query_data = datastore[query_image_key]
compare_results: Sequence[Tuple[float, float, float, str, Any]] = []

config["compare"]["filter_threshold"] = 200
#query_pose_lines = minmax_norm_by_imgrect(query_data["pose_lines"], query_data["width"], query_data["height"])
query_pose_lines_seq = norm_by_global_action(query_data["pose_lines"], query_data["global_action_lines"])
for target_key, target_data in zip(datastore.keys(), datastore.values()):
    if target_key == query_image_key:
        continue
    #combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, minmax_norm_by_imgrect(target_data["pose_lines"], target_data["width"], target_data["height"]))
    target_pose_lines_seq = norm_by_global_action(target_data["pose_lines"], target_data["global_action_lines"])
    pair_compare_results: Sequence[Tuple[float, float, float, str, Any]] = []
    for query_pose_lines in query_pose_lines_seq:
        for target_pose_lines in target_pose_lines_seq:
            combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, target_pose_lines)
            pair_compare_results.append((combined_ratio, hit_ratio, mean_distance_hits, target_key, target_data))
    compare_results.append(filter_pose_line_ga_result(pair_compare_results))
compare_results = np.array(compare_results)

# max_hit_ratio = max(compare_results[:,0]) # type: ignore
# only_max_res = compare_results[compare_results[:,0] >= max_hit_ratio] # type: ignore

# sorted_compare_results = compare_results[np.argsort(compare_results[:, 1])][::-1] # type: ignore => numpy two indice typing not fully supported
sorted_compare_results = compare_results[np.lexsort((compare_results[:,0], compare_results[:,1]))][::-1] # type: ignore


# sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])] # type: ignore => numpy two indice typing not fully supported

# plot and save the results:
fig, axis = plt.subplots(2, 6)

query_file_path = query_image_key # strip version number from key so we get filepath
query_img = download_img(query_data["row"])
query_img = converter.resize(query_img)
query_img = visualize.pose_lines(query_data["pose_lines"], query_img)
res_img = visualize.global_action_lines(query_data["global_action_lines"], query_img) # type: ignore
# visualize.safe("/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem/playground/query_img.jpg", query_img)

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

for res_idx, res in enumerate(sorted_compare_results[6:12]): #type: ignore
   #res_score, res_key, res_data = res # type: ignore => numpy unpack typing not fully supported
   combined_ratio, hit_ratio, mean_hit_dist, res_key, res_data = res # type: ignore => numpy unpack typing not fully supported
   #res_file_path = res_key[len(STORAGE_VERSION_KEY):] # strip version number from key so we get filepath
   #res_img = cv2.imread(res_file_path)
   res_img = download_img(res_data["row"])
   res_img = converter.resize(res_img)
   res_img = visualize.pose_lines(res_data["pose_lines"], res_img) # type: ignore
   res_img = visualize.global_action_lines(res_data["global_action_lines"], res_img) # type: ignore
   # visualize.safe("/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem/playground/res_img_"+str(res_idx+1)+".jpg", res_img)
   res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
   # matplot:
   ax = axis[1][res_idx] # type: ignore
   ax.imshow(res_img_rgb)
   ax.axis('off')
   ax.set_title("cr:{:.3f}\nhr:{:.3f}\nhd:{:.3f}\n{}".format(combined_ratio, hit_ratio, mean_hit_dist, res_data["row"]["class"]))

print(sorted_compare_results[0:5,-2]) #type: ignore

query_label = query_data["row"]["class"]
res_labels = list(map(lambda x: x["row"]["class"], sorted_compare_results[:,4]))
print(score_retrievals(query_label, res_labels))

plt.tight_layout()
plt.subplots_adjust(top=0.85) # Make space for title
plt.show()


"""
p@5 -> 2/5 = 0.4
p@10
"""