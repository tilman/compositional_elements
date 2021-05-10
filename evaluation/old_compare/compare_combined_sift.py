# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import datetime
import torch
from torch.functional import norm
from tqdm import tqdm
from . import eval_utils
import cv2

from .compare_deepfeatures import negative_cosine_dist_flatten

from compoelem.config import config
from compoelem.compare.pose_line import compare_pose_lines_2, compare_pose_lines_3, filter_pose_line_ga_result
from compoelem.compare.normalize import minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#brute-force-matching-with-sift-descriptors-and-ratio-test
def compare_siftBFMatcher1(sift1, sift2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    des1 = sift1["descriptors"]
    des2 = sift2["descriptors"]
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # see compare_orbBFMatcher2 for why we use len(matches) or len(good) from ratio test
    return len(good)/max(len(des1), len(des2))
    # => fmr: feature match ratio: zwischen 0 und 1. 0 => schlecht, 1 => guter match

def compare_combinedSetupB(data, sort_method):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        query_pose_lines_seq = norm_by_global_action(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["global_action_lines"])
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            target_pose_lines_seq = norm_by_global_action(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["global_action_lines"])
            pair_compare_results = []
            # include traditional features:
            fmr = compare_siftBFMatcher1(query_data["sift"], target_data["sift"])
            # feature match ratio

            for query_pose_lines in query_pose_lines_seq:
                for target_pose_lines in target_pose_lines_seq:
                    combined_ratio, hit_ratio, neg_mean_distance_hits = compare_pose_lines_3(query_pose_lines, target_pose_lines)
                    pair_compare_results.append((combined_ratio, hit_ratio, neg_mean_distance_hits, target_data))
            combined_ratio, hit_ratio, neg_mean_distance_hits, target_data = filter_pose_line_ga_result(pair_compare_results)
            fmrcr = fmr*combined_ratio
            fmrcr2 = fmr+combined_ratio
            compare_results.append((combined_ratio, hit_ratio, neg_mean_distance_hits, fmr, fmrcr, fmrcr2, target_data))
        compare_results = np.array(compare_results)
        sorted_compare_results = sort_method(compare_results)

        query_label = query_data["className"]
        res_labels = list(map(lambda x: x["className"], sorted_compare_results[:,-1]))
        metrics = eval_utils.score_retrievals(query_label, res_labels)
        label = metrics["label"]
        for key in metrics.keys():
            if key != "label":
                if key not in res_metrics:
                    res_metrics[key] = {}
                if label not in res_metrics[key]:
                    res_metrics[key][label] = []
                res_metrics[key][label].append(metrics[key])
    return eval_utils.get_eval_dataframe(res_metrics)

def lexsort_fmr_cr(compare_results):
    # (combined_ratio, hit_ratio, neg_mean_distance_hits, fmr, fmrcr, fmrcr2, target_data)
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,3], compare_results[:,0]))]
    return sorted_compare_results

def lexsort_fmr_hr(compare_results):
    # (combined_ratio, hit_ratio, neg_mean_distance_hits, fmr, fmrcr, fmrcr2, target_data)
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,3], compare_results[:,1]))]
    return sorted_compare_results

def lexsort_cr_fmr(compare_results):
    # (combined_ratio, hit_ratio, neg_mean_distance_hits, fmr, fmrcr, fmrcr2, target_data)
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,0], compare_results[:,3]))]
    return sorted_compare_results

def lexsort_hr_fmr(compare_results):
    # (combined_ratio, hit_ratio, neg_mean_distance_hits, fmr, fmrcr, fmrcr2, target_data)
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,1], compare_results[:,3]))]
    return sorted_compare_results

def sort_fmrcr1(compare_results):
    # (combined_ratio, hit_ratio, neg_mean_distance_hits, fmr, fmrcr, fmrcr2, target_data)
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 4])]
    return sorted_compare_results

def sort_fmrcr2(compare_results):
    # (combined_ratio, hit_ratio, neg_mean_distance_hits, fmr, fmrcr, fmrcr2, target_data)
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 5])]
    return sorted_compare_results

def eval_all_combinations(datastore, datastore_name):
    # TODO: quick and dirty code needs refactoring to look like compare_compoelem or compare_deepfeatures
    all_res_metrics = []
    for sort_method in [lexsort_fmr_cr, lexsort_fmr_hr, lexsort_cr_fmr, lexsort_hr_fmr, sort_fmrcr1, sort_fmrcr2]:
        start_time = datetime.datetime.now()
        experiment_id = "cB|"+sort_method.__name__+";A|ceb|normGlAC|th150;sift|bfm1"
        print("EXPERIMENT:", experiment_id)
        start_time = datetime.datetime.now()
        eval_dataframe = compare_combinedSetupB(list(datastore.values()), sort_method)
        all_res_metrics.append({
            "combinedSetup": "compare_combinedSetupB",
            "experiment_id": experiment_id,
            "sort_method": sort_method.__name__,
            "config": config,
            "datetime": start_time,
            "eval_time_s": (datetime.datetime.now() - start_time).seconds,
            "datastore_name": datastore_name,
            "eval_dataframe": eval_dataframe,
            "combined":True,
            "new": True,
        })
    return all_res_metrics
