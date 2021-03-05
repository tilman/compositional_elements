# call this script with `python -m evaluation.evaluate_poselines_globalaction`
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean, size
from compoelem.types import PoseLine
import os
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
from compoelem.compare.pose_line import compare_pose_lines_2
from compoelem.compare.normalize import minmax_norm_by_imgrect, minmax_norm_by_bbox
# from compoelem.detect.openpose.lib.utils.common import draw_humans
DATASTORE_FILE = os.path.join(os.path.dirname(__file__), "./datastore_icon_title_2.pkl")
datastore = pickle.load(open(DATASTORE_FILE, "rb"))

keys = list(datastore.keys())
keys.sort()
# keys = keys[0:10]

MAX_PER_CLASS = 10 # histogram only makes sense if we check exactly the same of each class. Since only 102 baptism images are avail. we have limit of 100

remaining_limits_global = {
    'annunciation': MAX_PER_CLASS, #383,
    'nativity': MAX_PER_CLASS, #163,
    'adoration': MAX_PER_CLASS, #418,
    'baptism': MAX_PER_CLASS, #102,
    'rape': MAX_PER_CLASS, #123,
    'virgin and child': MAX_PER_CLASS, #222,
}

config["compare"]["filter_threshold"]=0.2

hit_ratio_distribution = {}
mean_dist_distribution = {}
combinded_ratio_distribution = {}
for query_image_key in tqdm(keys, total=len(keys)):
    query_data = datastore[query_image_key]
    # query_pose_lines = query_data["pose_lines"]
    # query_pose_lines = minmax_norm_by_bbox(query_data["pose_lines"])
    query_pose_lines = minmax_norm_by_imgrect(query_data["pose_lines"], query_data["width"], query_data["height"])
    queryClassName = query_data["row"]["class"]
    
    if(queryClassName in remaining_limits_global and remaining_limits_global[queryClassName] > 0):
        remaining_limits_global[queryClassName] = remaining_limits_global[queryClassName] - 1
    else:
        continue
    if queryClassName not in hit_ratio_distribution:
        hit_ratio_distribution[queryClassName] = {}

    remaining_limits_local = {
        'annunciation': MAX_PER_CLASS, #383,
        'nativity': MAX_PER_CLASS, #163,
        'adoration': MAX_PER_CLASS, #418,
        'baptism': MAX_PER_CLASS, #102,
        'rape': MAX_PER_CLASS, #123,
        'virgin and child': MAX_PER_CLASS, #222,
    }

    for target_key in keys:
        target_data = datastore[target_key]
        targetClassName = target_data["row"]["class"]
        if(targetClassName in remaining_limits_local and remaining_limits_local[targetClassName] > 0):
            remaining_limits_local[targetClassName] = remaining_limits_local[targetClassName] - 1
        else:
            continue
        # combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, target_data["pose_lines"])
        # combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, minmax_norm_by_bbox(target_data["pose_lines"]))
        combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, minmax_norm_by_imgrect(target_data["pose_lines"], target_data["width"], target_data["height"]))
        if targetClassName in hit_ratio_distribution[queryClassName]:
            hit_ratio_distribution[queryClassName][targetClassName].append(hit_ratio)
        else:
            hit_ratio_distribution[queryClassName][targetClassName] = [hit_ratio]

keys_l1 = list(hit_ratio_distribution.keys())
fig, axis = plt.subplots(len(keys_l1))
for ik, key_l1 in enumerate(keys_l1):
    keys_l2 = list(hit_ratio_distribution[key_l1].keys())
    x = np.array(list(zip(*list(map(lambda x: hit_ratio_distribution[key_l1][x], keys_l2))))) # from dict to list
    n_bins = 10
    axis[ik].hist(x, n_bins, density=True, histtype='bar', label=keys_l2)
    axis[ik].legend(prop={'size': 10})
    axis[ik].set_title(key_l1)

# plt.hist(hit_ratio_distribution["adoration"],bins=10)
# plt.tight_layout()
# plt.subplots_adjust(top=0.85) # Make space for title
plt.show()