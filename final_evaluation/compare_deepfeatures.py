# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
import pandas as pd
import datetime
import torch
from tqdm import tqdm
from . import eval_utils

def eucl_dist_flatten(t1, t2):
    a = t1.detach().numpy().flatten()
    b = t2.detach().numpy().flatten()
    return np.linalg.norm(a-b)

def normal_cosine_dist(t1, t2):
    a = t1.detach().numpy().flatten()
    b = t2.detach().numpy().flatten()
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def negative_cosine_dist_flatten(t1, t2):
    return 1 - normal_cosine_dist(t1, t2)

def compare(compare_method_name, featuremap_key, data, sort_method):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            compare_results = compare_by_name(compare_method_name, featuremap_key, query_data, target_data, compare_results)
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

def compare_by_name(compare_method_name, featuremap_key, query_data, target_data, compare_results):
    if(compare_method_name == 'eucl_dist_flatten'):
        compare_results.append((eucl_dist_flatten(query_data[featuremap_key], target_data[featuremap_key]), target_data))
        return compare_results
    if(compare_method_name == 'negative_cosine_dist_flatten'):
        compare_results.append((negative_cosine_dist_flatten(query_data[featuremap_key], target_data[featuremap_key]), target_data))
        return compare_results
    raise NameError("provided compare_method_name does not exist")
    

def eval_all_combinations(datastore, datastore_name, featuremap_key):
    all_res_metrics = []
    for compare_method_name in ['eucl_dist_flatten', 'negative_cosine_dist_flatten']:
        start_time = datetime.datetime.now()
        experiment_id = "datastore: {}, featuremap_key: {}, compare_method: {}, sort_method: sort_distance_asc".format(datastore_name, featuremap_key, compare_method_name)
        print("EXPERIMENT:",experiment_id)
        eval_dataframe = compare(compare_method_name, featuremap_key, list(datastore.values()), sort_distance_asc)
        all_res_metrics.append({
            "experiment_id": experiment_id,
            "datetime": start_time,
            "eval_time_s": (datetime.datetime.now() - start_time).seconds,
            "datastore_name": datastore_name,
            "featuremap_key": featuremap_key,
            "compare_method": compare_method_name,
            "sort_method": "sort_distance_asc",
            "eval_dataframe": eval_dataframe,
            "new":True,
        })
    return all_res_metrics