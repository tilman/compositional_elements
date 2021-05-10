# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from . import eval_utils
import pickle
import copyreg
import cv2

from .compare_deepfeatures import negative_cosine_dist_flatten
from .compare_sift import compare_siftBFMatcher1

from compoelem.config import config
from compoelem.generate import global_action, pose_abstraction
from compoelem.compare.pose_line import compare_pose_lines_3, compare_pose_lines_3, filter_pose_line_ga_result
from compoelem.compare.normalize import minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action





# fix cv2 keypoint pickling error
def _pickle_keypoint(keypoint): #  : cv2.KeyPoint
    return cv2.KeyPoint, (
        keypoint.pt[0],
        keypoint.pt[1],
        keypoint.size,
        keypoint.angle,
        keypoint.response,
        keypoint.octave,
        keypoint.class_id,
    )
# Apply the bundling to pickle
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)





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







def compare_orbBFMatcher1(orb1, orb2):
    des1 = orb1["descriptors"]
    des2 = orb2["descriptors"]
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # see compare_orbBFMatcher2 for why we use len(matches) or len(good) from ratio test
    return len(matches)/max(len(des1), len(des2))

def compare_orbBFMatcher2(orb1, orb2):
    des1 = orb1["descriptors"]
    des2 = orb2["descriptors"]
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # INFO see https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    # -> Second param is boolean variable, crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets should match each other. It provides consistent result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.
    return len(matches) 






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

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#brute-force-matching-with-sift-descriptors-and-ratio-test
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
    # see compare_orbBFMatcher2 for why we use len(matches) or len(good) from ratio test
    return len(good)
    




    
def compare(data, compare_method, feature_key):
    res_metrics = {}
    precision_curves = {}
    all_retrieval_res = []
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            compare_results.append((compare_method(query_data[feature_key], target_data[feature_key]), target_data))
        compare_results = np.array(compare_results)
        sorted_compare_results = sort_desc(compare_results)
        query_label = query_data["className"]
        res_labels = list(map(lambda x: x["className"], sorted_compare_results[:,-1]))
        res_keys = list(map(lambda x: x["className"]+'_'+x["imgName"], sorted_compare_results[:,-1]))
        all_retrieval_res.append(np.array([
            query_data["className"]+'_'+query_data["imgName"],
            query_label,
            res_keys,
            res_labels
        ]))
        metrics = eval_utils.score_retrievals(query_label, res_labels)
        label = metrics["label"]
        if label in precision_curves:
            precision_curves[label].append(metrics["precision_at_rank"])
        else:
            precision_curves[label] = [metrics["precision_at_rank"]]
        for key in metrics.keys():
            if key != "label":
                if key not in res_metrics:
                    res_metrics[key] = {}
                if label not in res_metrics[key]:
                    res_metrics[key][label] = []
                res_metrics[key][label].append(metrics[key])
    return (eval_utils.get_eval_dataframe(res_metrics), precision_curves, np.array(all_retrieval_res))


def sort_desc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])][::-1]
    return sorted_compare_results





osuname = os.uname().nodename
print("osuname", osuname)
if osuname == 'MBP-von-Tilman' or osuname == 'MacBook-Pro-von-Tilman.local':
    COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
elif osuname == 'lme117':
    COMPOELEM_ROOT = "/home/zi14teho/compositional_elements"
else:
    COMPOELEM_ROOT = os.getenv('COMPOELEM_ROOT')
DATASTORE_NAME = "combined_datastore_ceb_dataset"
DATASTORE_FILE = COMPOELEM_ROOT+"/final_evaluation/"+DATASTORE_NAME+".pkl"
EVAL_RESULTS_FILE_DIR = COMPOELEM_ROOT+"/final_evaluation/final2pkl/"
DATASTORE_NAME = "combined_datastore_ceb_dataset"
datastore = pickle.load(open(DATASTORE_FILE, "rb"))
datastore_name = DATASTORE_NAME

# def eval_single_combination(
#         norm_method,
#         sort_method_name,
        
#         correction_angle,
#         cone_opening_angle,
#         cone_scale_factor,
#         cone_base_scale_factor,
#         filter_threshold,

#         poseline_fallback,
#         bisection_fallback,
#         glac_fallback,
#     ):

#     print({
#         "norm_method":norm_method,
#         "sort_method_name":sort_method_name,
#         "correction_angle":correction_angle,
#         "cone_opening_angle":cone_opening_angle,
#         "cone_scale_factor":cone_scale_factor,
#         "cone_base_scale_factor":cone_base_scale_factor,
#         "filter_threshold":filter_threshold,
#         "poseline_fallback":poseline_fallback,
#         "bisection_fallback":bisection_fallback,
#         "glac_fallback":glac_fallback,
#     })
def eval_single_combination(arg_obj):
    print(arg_obj)
    experiment_name = arg_obj["experiment_name"]
    compare_method_name = arg_obj["compare_method_name"]
    feature_key = arg_obj["feature_key"]
    
    if compare_method_name == "compare_briefBFMatcher1":
        compare_method = compare_briefBFMatcher1
    elif compare_method_name == "compare_briefBFMatcher2":
        compare_method = compare_briefBFMatcher2
    elif compare_method_name == "compare_orbBFMatcher1":
        compare_method = compare_orbBFMatcher1
    elif compare_method_name == "compare_orbBFMatcher2":
        compare_method = compare_orbBFMatcher2
    elif compare_method_name == "compare_siftBFMatcher1":
        compare_method = compare_siftBFMatcher1
    elif compare_method_name == "compare_siftBFMatcher2":
        compare_method = compare_siftBFMatcher2
    else:
        raise NotImplementedError("cmp not implemented")


    start_time = datetime.datetime.now()
    eval_dataframe, precision_curves, all_retrieval_res = compare(list(datastore.values()), compare_method, feature_key)
    filename = "final2_time{}_{}_{}.pkl".format(
        start_time.strftime("%d%m%y%H%M%S"),
        
        compare_method_name,
        feature_key,
    )
    print("filename", filename, "p@1", eval_dataframe["p@1"]["total (mean)"])
    res_summary = {
        "experiment_name": experiment_name,
        "experiment_id": filename,
        "filename": filename,
        "datetime": start_time,
        "eval_time_s": (datetime.datetime.now() - start_time).seconds,
        "datastore_name": datastore_name,

        "eval_dataframe": eval_dataframe,
        "precision_curves": precision_curves,
        "all_retrieval_res": all_retrieval_res,
        
        "config": config,

        "compare_method_name": compare_method_name,
        "feature_key": feature_key,
    }
    pickle.dump(res_summary, open(EVAL_RESULTS_FILE_DIR+filename, "wb"))
