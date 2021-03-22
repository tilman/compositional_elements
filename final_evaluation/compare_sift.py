# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
import cv2
import pandas as pd
import datetime
import torch
from tqdm import tqdm
from . import eval_utils

def compare_siftFLANN1(sift1, sift2):
    des1 = sift1["descriptors"]
    des2 = sift2["descriptors"]

    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks=30)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # return len(good)/max(len(des1, des2))
    return len(good)/max(len(des1), len(des2))

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
    # return len(good)/max(len(des1, des2))
    return len(good)/max(len(des1), len(des2))

def compare_siftBFMatcher2(sift1, sift2):
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
    # return len(good)/max(len(des1, des2))
    return len(good)
    
def compare(data, sort_method, compare_method):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            compare_results.append((compare_method(query_data["sift"], target_data["sift"]), target_data))
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
    for compare_method in [compare_siftBFMatcher1, compare_siftBFMatcher2]:
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
            "sift":True,
            "new":True,
        })
    return all_res_metrics