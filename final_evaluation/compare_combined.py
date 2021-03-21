# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import datetime
import torch
from torch.functional import norm
from tqdm import tqdm
from . import eval_utils

from .compare_deepfeatures import negative_cosine_dist_flatten

from compoelem.config import config
from compoelem.compare.pose_line import compare_pose_lines_2, filter_pose_line_ga_result
from compoelem.compare.normalize import minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action

def compare_combinedSetupA(data, sort_method):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        query_pose_lines_seq = norm_by_global_action(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["global_action_lines"])
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            target_pose_lines_seq = norm_by_global_action(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["global_action_lines"])
            pair_compare_results = []
            # include deep features:
            n_cos = negative_cosine_dist_flatten(query_data["imageNet_vgg19_bn_features"], target_data["imageNet_vgg19_bn_features"])

            for query_pose_lines in query_pose_lines_seq:
                for target_pose_lines in target_pose_lines_seq:
                    combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, target_pose_lines)
                    pair_compare_results.append((combined_ratio, hit_ratio, mean_distance_hits, target_data))
            combined_ratio, hit_ratio, neg_mean_distance_hits, target_data = filter_pose_line_ga_result(pair_compare_results)
            nccr = (n_cos/combined_ratio) if combined_ratio > 0 else 1/1000000
            nccr2 = n_cos*(1-combined_ratio)
            compare_results.append((combined_ratio, hit_ratio, neg_mean_distance_hits, n_cos, nccr, nccr2, target_data))
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

def lexsort_nccr_nc_hr_asc(compare_results):
    # (combined_ratio, hit_ratio, mean_distance_hits, n_cos, (n_cos/combined_ratio), target_data)
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,4], compare_results[:,3], compare_results[:,1]))]
    return sorted_compare_results

def lexsort_nc_hr_asc(compare_results):
    # (combined_ratio, hit_ratio, mean_distance_hits, n_cos, (n_cos/combined_ratio), target_data)
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,3], compare_results[:,1]))]
    return sorted_compare_results

def sort_ncos(compare_results):
    # (combined_ratio, hit_ratio, mean_distance_hits, n_cos, (n_cos/combined_ratio), target_data)
    # sorted_compare_results = compare_results[np.lexsort(compare_results[:,3])] #ncos3
    sorted_compare_results = compare_results[np.argsort(compare_results[:,3])] #ncos2
    return sorted_compare_results

# TODO
def sort_nccr(compare_results):
    # (combined_ratio, hit_ratio, mean_distance_hits, n_cos, (n_cos/combined_ratio), sort_nccr2, target_data)
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 4])]
    # sorted_compare_results = compare_results[np.argsort(compare_results[:,3])] #ncos2
    return sorted_compare_results

def sort_nccr2(compare_results):
    # (combined_ratio, hit_ratio, mean_distance_hits, n_cos, (n_cos/combined_ratio), sort_nccr2, target_data)
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 5])]
    # sorted_compare_results = compare_results[np.argsort(compare_results[:,3])] #ncos2
    return sorted_compare_results

#TODO
def lexsort_ncos_cr(compare_results):
    # (combined_ratio, hit_ratio, mean_distance_hits, n_cos, (n_cos/combined_ratio), target_data)
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,3], compare_results[:,0]))]
    return sorted_compare_results

def eval_all_combinations(datastore, datastore_name):
    # TODO: quick and dirty code needs refactoring to look like compare_compoelem or compare_deepfeatures
    all_res_metrics = []
    start_time = datetime.datetime.now()
    experiment_id = "cA|sortNcCr;A|ceb|normGlAC|th150;img_vggBn"
    print("EXPERIMENT:", experiment_id)
    start_time = datetime.datetime.now()
    eval_dataframe = compare_combinedSetupA(list(datastore.values()), sort_nccr)
    all_res_metrics.append({
        "experiment_id": experiment_id,
        "config": config,
        "datetime": start_time,
        "eval_time_s": (datetime.datetime.now() - start_time).seconds,
        "datastore_name": datastore_name,
        "eval_dataframe": eval_dataframe,
        "new": True,
    })
    # experiment_id = "cA|sortNcHr;A|ceb|normGlAC|th150;img_vggBn|ncos"
    # print("EXPERIMENT:", experiment_id)
    # start_time = datetime.datetime.now()
    # eval_dataframe = compare_combinedSetupA(list(datastore.values()), lexsort_nc_hr_asc)
    # all_res_metrics.append({
    #     "experiment_id": experiment_id,
    #     "config": config,
    #     "datetime": start_time,
    #     "eval_time_s": (datetime.datetime.now() - start_time).seconds,
    #     "datastore_name": datastore_name,
    #     "eval_dataframe": eval_dataframe,
    #     "new": True,
    # })
    # experiment_id = "cA|sortNccrNcHr;A|ceb|normGlAC|th150;img_vggBn|ncos"
    # print("EXPERIMENT:", experiment_id)
    # start_time = datetime.datetime.now()
    # eval_dataframe = compare_combinedSetupA(list(datastore.values()), lexsort_nccr_nc_hr_asc)
    # all_res_metrics.append({
    #     "experiment_id": experiment_id,
    #     "config": config,
    #     "datetime": start_time,
    #     "eval_time_s": (datetime.datetime.now() - start_time).seconds,
    #     "datastore_name": datastore_name,
    #     "eval_dataframe": eval_dataframe,
    #     "new": True,
    # })
    return all_res_metrics
