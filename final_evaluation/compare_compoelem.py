# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
import pandas as pd
import datetime
import torch
from tqdm import tqdm
from . import eval_utils

from compoelem.config import config
from compoelem.compare.pose_line import compare_pose_lines_2, filter_pose_line_ga_result
from compoelem.compare.normalize import minmax_norm_by_imgrect, norm_by_global_action

def compare_setupA(data):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        #query_pose_lines = minmax_norm_by_imgrect(query_data["compoelem"]["pose_lines"], query_data["width"], query_data["height"])
        query_pose_lines_seq = norm_by_global_action(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["global_action_lines"])
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            #combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, minmax_norm_by_imgrect(target_data["compoelem"]["pose_lines"], target_data["width"], target_data["height"]))
            target_pose_lines_seq = norm_by_global_action(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["global_action_lines"])
            pair_compare_results = []
            for query_pose_lines in query_pose_lines_seq:
                for target_pose_lines in target_pose_lines_seq:
                    combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_2(query_pose_lines, target_pose_lines)
                    pair_compare_results.append((combined_ratio, hit_ratio, mean_distance_hits, target_data))
            compare_results.append(filter_pose_line_ga_result(pair_compare_results))
        compare_results = np.array(compare_results)
        # max_hit_ratio = max(compare_results[:,0]) # type: ignore
        # only_max_res = compare_results[compare_results[:,0] >= max_hit_ratio] # type: ignore
        # sorted_compare_results = compare_results[np.argsort(compare_results[:, 1])][::-1] # type: ignore => numpy two indice typing not fully supported
        sorted_compare_results = compare_results[np.lexsort((compare_results[:,1], compare_results[:,2]))][::-1] # 0,1 -> level1:hit_ratio, level2:mean_distance_hits
        #sorted_compare_results = compare_results[np.lexsort((compare_results[:,0], compare_results[:,1]))][::-1] # 0,1 -> level1:combined_ratio, level2:hit_ratio
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
    avgerave_metrics = {}
    for metricKey in res_metrics.keys():
        if metricKey != "label":
            if metricKey not in avgerave_metrics:
                avgerave_metrics[metricKey] = {}
            total_list = []
            for label in res_metrics[metricKey].keys():
                avgerave_metrics[metricKey][label] = np.mean(res_metrics[metricKey][label]) # mean for each class
                total_list.append(res_metrics[metricKey][label])
            avgerave_metrics[metricKey]["total (mean)"] = np.mean(list(avgerave_metrics[metricKey].values())) # mean of all classes means
            avgerave_metrics[metricKey]["total (w. mean)"] = np.mean(np.array(total_list).flatten()) # mean of all values regardless of class (-> the same as class mean weighted by amount of datapoints in class)
    return pd.DataFrame(avgerave_metrics)

def sort_distance_asc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:,0])]
    return sorted_compare_results

def eval_all_combinations(datastore, datastore_name):
    all_res_metrics = []
    # for compare_method_name in ['eucl_dist_flatten', 'negative_cosine_dist_flatten']:
    for filter_threshold in [100, 150, 175, 200, 250]:
        config["compare"]["filter_threshold"] = filter_threshold
        start_time = datetime.datetime.now()
        #experiment_id = "datastore: {}, setup: A, filter_threshold: {}, normalisation: norm_by_global_action, compare_method: compare_pose_lines_2, result_filter_method: filter_pose_line_ga_result, sort_method: lexsort[hit_ratio, mean_distance_hits]".format(datastore_name, filter_threshold)
        experiment_id = "datastore: {}, setup: A, filter_threshold: {}, normalisation: norm_by_global_action, compare_method: compare_pose_lines_2, result_filter_method: filter_pose_line_ga_result, sort_method: lexsort[combined_ratio, hit_ratio]".format(datastore_name, filter_threshold)
        print("EXPERIMENT:",experiment_id)
        eval_dataframe = compare_setupA(list(datastore.values()))
        all_res_metrics.append({
            "experiment_id": experiment_id,
            "config": config,
            "datetime": start_time,
            "eval_time_s": (datetime.datetime.now() - start_time).seconds,
            "datastore_name": datastore_name,
            "normalisation": "norm_by_global_action",
            "compare_method": "compare_pose_lines_2",
            "result_filter_method": "filter_pose_line_ga_result",
            #"sort_method": "lexsort[hit_ratio, mean_distance_hits]",
            "sort_method": "lexsort[combined_ratio, hit_ratio]",
            "eval_dataframe": eval_dataframe,
            "new":True,
        })
    return all_res_metrics