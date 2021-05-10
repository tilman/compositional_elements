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





def eucl_dist_flatten(t1, t2):
    a = t1.detach().numpy().flatten()
    b = t2.detach().numpy().flatten()
    return np.linalg.norm(a-b)

def normal_cosine_dist(t1, t2):
    a = t1.detach().numpy().flatten()
    b = t2.detach().numpy().flatten()
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def negative_cosine_dist_flatten(t1, t2):
    """returns negative cosine similarity between two pytorch tensors. 0 => exatly the same, 2 => exactly oposite

    Explanaition:
    => cosine sim: −1 meaning exactly opposite, to 1 meaning exactly the same, with 0
    => negative cosine sim = 1 - cosine sim

    Args:
        t1 ([type]): [description]
        t2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 1 - normal_cosine_dist(t1, t2)



# TODO: extend eval res for qualitative plotting:
'''
# first step, find best result and store index for it
res_classNames = [
    # 25k string variables
    res_for_key_idx1: ["baptism","baptism",...,"aduration"]
]
# second step, find indices for best result
res_keys = [
    # 25k string variables
    res_for_key_idx1: ["key_for_rank1","key_for_rank2",...,"key_for_rank50"]
    ...
    res_for_key_idx500:
]
res = [
    {query_key, query_label, res_keys, res_labels}
]
'''

    
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
        sorted_compare_results = sort_asc(compare_results)
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


def sort_asc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:, 0])]
    return sorted_compare_results





osuname = os.uname().nodename
print("osuname 2", osuname, "__name__",__name__)
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
    print("arg_obj", arg_obj)
    experiment_name = arg_obj["experiment_name"]
    compare_method_name = arg_obj["compare_method_name"]
    feature_key = arg_obj["feature_key"]
    
    if compare_method_name == "eucl_dist_flatten":
        compare_method = eucl_dist_flatten
    elif compare_method_name == "negative_cosine_dist_flatten":
        compare_method = negative_cosine_dist_flatten
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
        
        "compare_method_name": compare_method_name,
        "feature_key": feature_key,
    }
    pickle.dump(res_summary, open(EVAL_RESULTS_FILE_DIR+filename, "wb"))
