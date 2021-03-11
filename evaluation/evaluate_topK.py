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
            "p@50": -1,
            "p@rel": -1,
            "mAP": -1,
            "r@1": -1,
            "r@5": -1,
            "r@10": -1,
            "r@50": -1,
            "r@rel": -1,
            "mAR": -1
        }
        return metrics
    # computing precision based metrics
    precision_at_rank = np.cumsum(relevant_mask) / np.arange(1, len(relevant_mask) + 1)
    precision_at_1 = precision_at_rank[0]
    precision_at_5 = precision_at_rank[4]
    precision_at_10 = precision_at_rank[9]
    precision_at_50 = precision_at_rank[49]
    precision_at_rel = precision_at_rank[num_relevant_retrievals - 1]
    average_precision = np.sum(precision_at_rank * relevant_mask) / num_relevant_retrievals
    # computing recall based metrics
    recall_at_rank = np.cumsum(relevant_mask) / num_relevant_retrievals
    recall_at_1 = recall_at_rank[0]
    recall_at_5 = recall_at_rank[4]
    recall_at_10 = recall_at_rank[9]
    recall_at_50 = recall_at_rank[49]
    recall_at_rel = recall_at_rank[num_relevant_retrievals - 1]
    average_recall = np.sum(recall_at_rank * relevant_mask) / num_relevant_retrievals
    metrics = {
        "label": label,
        "p@1": precision_at_1,
        "p@5": precision_at_5,
        "p@10": precision_at_10,
        "p@10": precision_at_50,
        "p@rel": precision_at_rel,
        "mAP": average_precision,
        "r@1": recall_at_1,
        "r@5": recall_at_5,
        "r@10": recall_at_10,
        "r@10": recall_at_50,
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
DATASET = "rebalanced_datastore_icon_title_2_noRape"
DATASTORE_FILE = os.path.join(os.path.dirname(__file__), "../"+DATASET+".pkl")
datastore = pickle.load(open(DATASTORE_FILE, "rb"))

for threshold in [100, 150, 175, 200, 250]:
    res_metrics = {}
    config["compare"]["filter_threshold"] = threshold
    for query_image_key in tqdm(list(datastore.keys()), total=len(datastore.keys())):
        query_data = datastore[query_image_key]
        compare_results: Sequence[Tuple[float, float, float, str, Any]] = []
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
        query_label = query_data["row"]["class"]
        res_labels = list(map(lambda x: x["row"]["class"], sorted_compare_results[:,4]))
        metrics = score_retrievals(query_label, res_labels)
        label = metrics["label"]
        for key in metrics.keys():
            if key != "label":
                if key not in res_metrics:
                    res_metrics[key] = {}
                if label not in res_metrics[key]:
                    res_metrics[key][label] = []
                res_metrics[key][label].append(metrics[key])
        #print(metrics)
        # "label": label,
        # "p@1": precision_at_1,
        # "p@5": precision_at_5,
        # "p@10": precision_at_10,
        # "p@rel": precision_at_rel,
        # "mAP": average_precision,
        # "r@1": recall_at_1,
        # "r@5": recall_at_5,
        # "r@10": recall_at_10,
        # "r@rel": recall_at_rel,
        # "mAR": average_recall
    #print(np.mean(res_metrics["p@1"]["annunciation"]), np.mean(res_metrics["p@5"]["annunciation"]), np.mean(res_metrics["p@10"]["annunciation"]))
    print('\n')
    print("# "+DATASET)
    print('config["compare"]["filter_threshold"] = ', threshold)
    print()
    print("| class | "+" | ".join(res_metrics.keys())+" |")
    print("| - | "+" | ".join(["-"] * len(res_metrics.keys()))+" |")
    for cn in ['annunciation', 'nativity', 'adoration', 'baptism', 'virgin and child']:
        means = []
        for km in res_metrics.keys():
            means.append("{:.4f}".format(np.mean(res_metrics[km][cn])))
        print("| "+cn+" | "+" | ".join(means)+" |")
    print("| total | "+" | ".join(["{:.4f}".format(np.mean(np.array(list(m.values())).flatten())) for m in res_metrics.values()])+" |")