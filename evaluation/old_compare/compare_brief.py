# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
import cv2
import pandas as pd
import datetime
import torch
from tqdm import tqdm
from . import eval_utils

def compare_briefBFMatcher1(brief1, brief2):
    des1 = brief1["descriptors"]
    des2 = brief2["descriptors"]
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # see compare_orbBFMatcher2 for why we use len(matches) or len(good) from ratio test
    return len(matches)/max(len(des1), len(des2))

def compare_briefBFMatcher2(brief1, brief2):
    des1 = brief1["descriptors"]
    des2 = brief2["descriptors"]
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # see compare_orbBFMatcher2 for why we use len(matches) or len(good) from ratio test
    return len(matches)
    
def compare(data, sort_method, compare_method):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            compare_results.append((compare_method(query_data["brief"], target_data["brief"]), target_data))
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


def sort_desc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])][::-1]
    return sorted_compare_results

def eval_all_combinations(datastore, datastore_name):
    all_res_metrics = []
    for compare_method in [compare_briefBFMatcher1, compare_briefBFMatcher2]:
        start_time = datetime.datetime.now()
        sortmethod = sort_desc
        experiment_id = "datastore: {}, compare_method: {}, sort_method: {}".format(datastore_name, compare_method.__name__, sortmethod.__name__)
        print("EXPERIMENT:",experiment_id)
        eval_dataframe = compare(list(datastore.values()), sortmethod, compare_method)
        all_res_metrics.append({
            "experiment_id": experiment_id,
            "datetime": start_time,
            "eval_time_s": (datetime.datetime.now() - start_time).seconds,
            "datastore_name": datastore_name,
            "compare_method": compare_method.__name__,
            "sort_method": sortmethod.__name__,
            "eval_dataframe": eval_dataframe,
            "brief":True,
            "new":True,
        })
    return all_res_metrics