# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import datetime
import torch
from torch.functional import norm
from tqdm import tqdm
from . import eval_utils

from compoelem.config import config
from compoelem.compare.pose_line import compare_pose_lines_3, compare_pose_lines_3, filter_pose_line_ga_result
from compoelem.compare.normalize import minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action

def compare_setupA(data, sort_method, norm_method):
    if norm_method != 'norm_by_global_action':
        raise NotImplementedError("only norm_by_global_action is implemented")
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        #query_pose_lines = minmax_norm_by_imgrect(query_data["compoelem"]["pose_lines"], query_data["width"], query_data["height"])
        query_pose_lines_seq = norm_by_global_action(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["global_action_lines"])
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            #combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_3(query_pose_lines, minmax_norm_by_imgrect(target_data["compoelem"]["pose_lines"], target_data["width"], target_data["height"]))
            target_pose_lines_seq = norm_by_global_action(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["global_action_lines"])
            pair_compare_results = []
            for query_pose_lines in query_pose_lines_seq:
                for target_pose_lines in target_pose_lines_seq:
                    combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_3(query_pose_lines, target_pose_lines)
                    pair_compare_results.append((combined_ratio, hit_ratio, mean_distance_hits, target_data))
            compare_results.append(filter_pose_line_ga_result(pair_compare_results))
        compare_results = np.array(compare_results)
        # max_hit_ratio = max(compare_results[:,0]) # type: ignore
        # only_max_res = compare_results[compare_results[:,0] >= max_hit_ratio] # type: ignore
        # sorted_compare_results = compare_results[np.argsort(compare_results[:, 1])][::-1] # type: ignore => numpy two indice typing not fully supported
        #sorted_compare_results = compare_results[np.lexsort((compare_results[:,1], compare_results[:,2]))][::-1] # 0,1 -> level1:hit_ratio, level2:mean_distance_hits
        #sorted_compare_results = compare_results[np.lexsort((compare_results[:,0], compare_results[:,1]))][::-1] # 0,1 -> level1:combined_ratio, level2:hit_ratio
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

def compare_setupB(data, sort_method, norm_method):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        if norm_method == 'minmax_norm_by_imgrect':
            query_pose_lines = minmax_norm_by_imgrect(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["width"], query_data["compoelem"]["height"])
        elif norm_method == 'minmax_norm_by_bbox':
            query_pose_lines = minmax_norm_by_bbox(query_data["compoelem"]["pose_lines"])
        else:
            raise NotImplementedError("norm_method: {} not implemented".format(norm_method))
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            #combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_3(query_pose_lines, minmax_norm_by_imgrect(target_data["compoelem"]["pose_lines"], target_data["width"], target_data["height"]))

            if norm_method == 'minmax_norm_by_imgrect':
                target_pose_lines = minmax_norm_by_imgrect(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["width"], target_data["compoelem"]["height"])
            elif norm_method == 'minmax_norm_by_bbox':
                target_pose_lines = minmax_norm_by_bbox(target_data["compoelem"]["pose_lines"])
            else:
                raise NotImplementedError("norm_method: {} not implemented".format(norm_method))
            combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_3(query_pose_lines, target_pose_lines)
            compare_results.append((combined_ratio, hit_ratio, mean_distance_hits, target_data))
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

def lexsort_hr_md(compare_results):
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,1], compare_results[:,2]))][::-1] # 0,1 -> level1:hit_ratio, level2:mean_distance_hits
    return sorted_compare_results

def lexsort_cr_hr(compare_results):
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,0], compare_results[:,1]))][::-1] # 0,1 -> level1:combined_ratio, level2:hit_ratio
    return sorted_compare_results

def eval_all_combinations(datastore, datastore_name):
    all_res_metrics = []
    for sort_method in [lexsort_hr_md, lexsort_cr_hr]:
        # for setup in [compare_setupA, compare_setupB]:
        for setup in [compare_setupA]:
            for norm_method in ['minmax_norm_by_imgrect', 'minmax_norm_by_bbox'] if setup.__name__ == 'compare_setupB' else ['norm_by_global_action']:
                # for filter_threshold in [75, 100, 125, 150, 175, 200]:
                for filter_threshold in [150]:
                    config["compare"]["filter_threshold"] = filter_threshold
                    start_time = datetime.datetime.now()
                    if setup.__name__ == 'compare_setupA':
                        result_filter_method_name = "filter_pose_line_ga_result"
                    elif setup.__name__ == 'compare_setupB':
                        result_filter_method_name = "none"
                    experiment_id = "datastore: {}, setup: {}, filter_threshold: {}, norm_method: {}, compare_method: compare_pose_lines_3, result_filter_method: {}, sort_method: {}".format(
                        datastore_name, setup.__name__, filter_threshold, norm_method, result_filter_method_name, sort_method.__name__
                    )
                    print("EXPERIMENT:", experiment_id)
                    start_time = datetime.datetime.now()
                    eval_dataframe = setup(list(datastore.values()), sort_method, norm_method)
                    all_res_metrics.append({
                        "experiment_id": experiment_id,
                        "config": config,
                        "filter_threshold": filter_threshold,
                        "datetime": start_time,
                        "setup": setup.__name__,
                        "eval_time_s": (datetime.datetime.now() - start_time).seconds,
                        "datastore_name": datastore_name,
                        "norm_method": norm_method,
                        "compare_method": "compare_pose_lines_3",
                        "result_filter_method": result_filter_method_name,
                        "sort_method": sort_method.__name__,
                        "eval_dataframe": eval_dataframe,
                        "new": True,
                    })
    return all_res_metrics
