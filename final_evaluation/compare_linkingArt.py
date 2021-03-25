# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
import cv2
import pandas as pd
import datetime
import torch
from tqdm import tqdm
from . import eval_utils
from compoelem.detect.openpose.lib.utils.common import BodyPart, Human, CocoPart

def neg_cos_dist(r_tick, s_tick):
    a = r_tick.flatten()
    b = s_tick.flatten()
    return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) # type: ignore, manually checked it works!

def flipped_cosine_min_dist(r_tick, s_tick):
    s_star = np.array([[-kp[0], kp[1]] for kp in s_tick]) #TODO check if correct
    return min(
        neg_cos_dist(r_tick, s_tick),
        neg_cos_dist(r_tick, s_star),
    )

def openpose_to_nparray(human: Human):
    keypoints = [
        [human.body_parts[i].x, human.body_parts[i].y] if i in human.body_parts else np.array([0,0]) for i in range(0, 18)
    ]
    return np.array(keypoints)

def isNoneKp(kp):
    return kp[0] == 0 and kp[1] == 0

def neck_norm_poses(r, s): # TODO check
    ROOT_POINT = CocoPart.Neck.value
    r_root = r[ROOT_POINT]
    s_root = s[ROOT_POINT]
    if(isNoneKp(r_root) or isNoneKp(s_root)):
        raise ValueError("neck point missing, normalization not possible, skipping that pose")
    r_tick = []
    s_tick = []
    for r_i, s_i in zip(r, s):
        if(not isNoneKp(r_i) or not isNoneKp(s_i)): # if i€I_r,s
            r_tick.append(r_i - r_root)
            s_tick.append(s_i - s_root)
        else: # else case 
            r_tick.append(np.array([0, 0]))
            s_tick.append(np.array([0, 0]))
    return np.array(r_tick), np.array(s_tick)

def compare_dist_min(poses_i1, poses_i2): #in paper this is dist_min(i1, i2), we do not input images but rather input the precomputed poses directly
    poses_i1 = np.array([openpose_to_nparray(human) for human in poses_i1]) # output shape of each item is (18, 2) since we are using the 18 openpose keypoint model
    poses_i2 = np.array([openpose_to_nparray(human) for human in poses_i2])
    dist = []
    combinations = []
    for idx_r, r in enumerate(poses_i1):
        for idx_s, s in enumerate(poses_i2):
            try:
                r_tick, s_tick = neck_norm_poses(r, s)
            except ValueError as e: # "neck point missing, normalization not possible, skipping that pose"  => this edge case is not mentioned in the paper but was the only sensible decision I think
                #print(e) 
                continue
            dist.append(flipped_cosine_min_dist(r_tick, s_tick))
            combinations.append((idx_r, idx_s))
    if(len(dist) == 0):
        return (2, (0,0)) #maximum possible neg cos dist
    else:
        # return min(dist)
        am = np.argmin(np.array(dist))
        return (dist[am], combinations[am])
    
def compare(data, sort_method, compare_method):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            distance, min_combination = compare_method(query_data["compoelem"]["humans"], target_data["compoelem"]["humans"])
            compare_results.append((distance, min_combination, target_data)) # we will use the same precomputed poses as we already computed for compoelem method
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
    eval_dataframe = pd.DataFrame(avgerave_metrics)
    print(eval_dataframe)
    return eval_dataframe


def sort_desc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])][::-1]
    return sorted_compare_results

def sort_asc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])]
    return sorted_compare_results

def eval_all_combinations(datastore, datastore_name):
    all_res_metrics = []
    for compare_method in [compare_dist_min]:
        start_time = datetime.datetime.now()
        sortmethod = sort_asc
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
            "linkingArt":True,
            "new":True,
        })
    return all_res_metrics