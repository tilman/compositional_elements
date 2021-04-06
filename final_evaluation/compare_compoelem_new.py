# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import datetime
import torch
import pickle
from torch.functional import norm
from tqdm import tqdm
from . import eval_utils

from compoelem.config import config
from compoelem.generate import global_action, pose_abstraction
from compoelem.compare.pose_line import compare_pose_lines_3, compare_pose_lines_3, filter_pose_line_ga_result
from compoelem.compare.normalize import minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action

def compare_setupA(data, sort_method, norm_method):
    if norm_method != 'norm_by_global_action':
        raise NotImplementedError("only norm_by_global_action is implemented")
    res_metrics = {}
    precision_curves = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        #query_pose_lines = minmax_norm_by_imgrect(query_data["compoelem"][pose_lines_key], query_data["width"], query_data["height"])
        query_pose_lines_seq = norm_by_global_action(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["global_action_lines"])
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            #combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_3(query_pose_lines, minmax_norm_by_imgrect(target_data["compoelem"][pose_lines_key], target_data["width"], target_data["height"]))
            target_pose_lines_seq = norm_by_global_action(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["global_action_lines"])
            pair_compare_results = []
            for query_pose_lines in query_pose_lines_seq:
                for target_pose_lines in target_pose_lines_seq:
                    combined_ratio, hit_ratio, mean_distance_hits = compare_pose_lines_3(query_pose_lines, target_pose_lines)
                    pair_compare_results.append((combined_ratio, hit_ratio, mean_distance_hits, target_data))
            compare_results.append(filter_pose_line_ga_result(pair_compare_results))
        compare_results = np.array(compare_results)
        sorted_compare_results = sort_method(compare_results)
        query_label = query_data["className"]
        res_labels = list(map(lambda x: x["className"], sorted_compare_results[:,-1]))
        metrics = eval_utils.score_retrievals(query_label, res_labels)
        label = metrics["label"]
        precision_curves[label] = metrics["precision_at_rank"]
        for key in metrics.keys():
            if key != "label":
                if key not in res_metrics:
                    res_metrics[key] = {}
                if label not in res_metrics[key]:
                    res_metrics[key][label] = []
                res_metrics[key][label].append(metrics[key])
    return (eval_utils.get_eval_dataframe(res_metrics), precision_curves)

def lexsort_cr_hr(compare_results):
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,0], compare_results[:,1]))][::-1] # 0,1 -> level1:hit_ratio, level2:combined_ratio, 
    return sorted_compare_results

# def eval_all_combinations(datastore, datastore_name):
def eval_all_combinations(datastore, datastore_name, filter_threshold, with_fallback):
    tmp_eval_log = []
    all_res_metrics = []
    sort_method = lexsort_cr_hr
    setup = compare_setupA
    norm_method = 'norm_by_global_action'
    for cone_base_scale_factor in [0, 1, 2, 2.5]:
        for cone_scale_factor in [5, 10, 15]:
            for cone_opening_angle in [70, 80, 90]:
                for correction_angle in [40, 50]:
                    config["bisection"]["cone_base_scale_factor"] = cone_base_scale_factor
                    config["bisection"]["correction_angle"] = correction_angle
                    config["bisection"]["cone_opening_angle"] = cone_opening_angle
                    config["bisection"]["cone_scale_factor"] = cone_scale_factor
                    config["compare"]["filter_threshold"] = filter_threshold
                    new_datastore_values = []
                    for key in datastore.keys():
                        poses = datastore[key]["compoelem"]["poses"]
                        datastore[key]["compoelem"]["global_action_lines"] = global_action.get_global_action_lines(poses, fallback=with_fallback)
                        datastore[key]["compoelem"]["pose_lines"] = pose_abstraction.get_pose_lines(poses, fallback=with_fallback)
                        new_datastore_values.append(datastore[key])
                    start_time = datetime.datetime.now()
                    if setup.__name__ == 'compare_setupA':
                        result_filter_method_name = "filter_pose_line_ga_result"
                    else:
                        result_filter_method_name = "none"
                    experiment_id = "datastore: {}, setup: {}, filter_threshold: {}, norm_method: {}, compare_method: compare_pose_lines_3, result_filter_method: {}, sort_method: {}, correction_angle: {}, cone_opening_angle: {}, cone_scale_factor: {}".format(
                        datastore_name, setup.__name__, filter_threshold, norm_method, result_filter_method_name, sort_method.__name__, correction_angle, cone_opening_angle, cone_scale_factor
                    )
                    print("EXPERIMENT:", experiment_id)
                    start_time = datetime.datetime.now()
                    eval_dataframe, precision_curves = setup(list(datastore.values()), sort_method, norm_method)
                    all_res_metrics.append({
                        "experiment_id": experiment_id,
                        "config": config,
                        "filter_threshold": filter_threshold,
                        "correction_angle": correction_angle,
                        "cone_opening_angle": cone_opening_angle,
                        "cone_scale_factor": cone_scale_factor,
                        "cone_base_scale_factor": cone_base_scale_factor,
                        "datetime": start_time,
                        "setup": setup.__name__,
                        "eval_time_s": (datetime.datetime.now() - start_time).seconds,
                        "datastore_name": datastore_name,
                        "norm_method": norm_method,
                        "compare_method": "compare_pose_lines_3",
                        "result_filter_method": result_filter_method_name,
                        "sort_method": sort_method.__name__,
                        "eval_dataframe": eval_dataframe,
                        "precision_curves": precision_curves,
                        "with_fallback": with_fallback,
                        "new": True,
                    })
                    pickle.dump(tmp_eval_log, open(".tmpEvalLog_fth{}_{}".format(filter_threshold,"fb" if with_fallback else "noFb"), "wb"))
    return all_res_metrics
