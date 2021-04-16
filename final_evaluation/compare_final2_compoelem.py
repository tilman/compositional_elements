# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import os
import numpy as np
import datetime
from tqdm import tqdm
from . import eval_utils
import pickle
import copyreg
import cv2

from .compare_deepfeatures import negative_cosine_dist_flatten, eucl_dist_flatten
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

def compare_setupA(data, sort_method, norm_method, glac_fallback, compare_other, additional_feature_weight):
    if norm_method != 'norm_by_global_action':
        raise NotImplementedError("only norm_by_global_action is implemented")
    res_metrics = {}
    precision_curves = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        #query_pose_lines = minmax_norm_by_imgrect(query_data["compoelem"][pose_lines], query_data["width"], query_data["height"])
        query_pose_lines_seq = norm_by_global_action(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["global_action_lines"], fallback=glac_fallback)
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            if compare_other == 'vgg19_ncos':
                r_addition = negative_cosine_dist_flatten(query_data["imageNet_vgg19_bn_features"], target_data["imageNet_vgg19_bn_features"])
            elif compare_other == 'resnet50_cos':
                r_addition = negative_cosine_dist_flatten(query_data["places365_resnet50_feature_noFC"], target_data["places365_resnet50_feature_noFC"])
            elif compare_other == 'resnet50_eucl':
                r_addition = eucl_dist_flatten(query_data["places365_resnet50_feature_noFC"], target_data["places365_resnet50_feature_noFC"])
            elif compare_other == 'sift_bfm1':
                r_addition = compare_siftBFMatcher1(query_data["sift"], target_data["sift"])
            elif compare_other is None:
                r_addition = 0
            else:
                raise NotImplementedError("not implemented compare_other", compare_other)

            #combined_ratio, hit_ratio, neg_mean_distance_hits = compare_pose_lines_3(query_pose_lines, minmax_norm_by_imgrect(target_data["compoelem"][pose_lines], target_data["width"], target_data["height"]))
            target_pose_lines_seq = norm_by_global_action(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["global_action_lines"], fallback=glac_fallback)
            pair_compare_results = []
            for query_pose_lines in query_pose_lines_seq:
                for target_pose_lines in target_pose_lines_seq:
                    combined_ratio, hit_ratio, neg_mean_distance_hits = compare_pose_lines_3(query_pose_lines, target_pose_lines)
                    pair_compare_results.append((combined_ratio, hit_ratio, neg_mean_distance_hits, target_data))
            combined_ratio, hit_ratio, neg_mean_distance_hits, target_data = filter_pose_line_ga_result(pair_compare_results)

            a = additional_feature_weight
            wra = r_addition * (1-a)
            r_combi1 = wra * (1 - combined_ratio * a)
            r_combi2 = wra + (1 - combined_ratio * a)
            r_combi3 = wra * (1 - neg_mean_distance_hits * a)
            r_combi4 = wra + (1 - neg_mean_distance_hits * a)


            compare_results.append((combined_ratio, hit_ratio, neg_mean_distance_hits, r_combi1, r_combi2, r_combi3, r_combi4, r_addition, target_data))
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

def compare_setupB(data, sort_method, norm_method, glac_fallback, compare_other, additional_feature_weight):
    if compare_other is not None:
        raise NotImplementedError("compare other not implemented")
    res_metrics = {}
    precision_curves = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        if norm_method == 'none':
            query_pose_lines = query_data["compoelem"]["pose_lines"]
        elif norm_method == 'minmax_norm_by_imgrect':
            query_pose_lines = minmax_norm_by_imgrect(query_data["compoelem"]["pose_lines"], query_data["compoelem"]["width"], query_data["compoelem"]["height"])
        elif norm_method == 'minmax_norm_by_bbox':
            query_pose_lines = minmax_norm_by_bbox(query_data["compoelem"]["pose_lines"])
        else:
            raise NotImplementedError("norm_method: {} not implemented".format(norm_method))
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            if norm_method == 'none':
                target_pose_lines = target_data["compoelem"]["pose_lines"]
            elif norm_method == 'minmax_norm_by_imgrect':
                target_pose_lines = minmax_norm_by_imgrect(target_data["compoelem"]["pose_lines"], target_data["compoelem"]["width"], target_data["compoelem"]["height"])
            elif norm_method == 'minmax_norm_by_bbox':
                target_pose_lines = minmax_norm_by_bbox(target_data["compoelem"]["pose_lines"])
            else:
                raise NotImplementedError("norm_method: {} not implemented".format(norm_method))
            combined_ratio, hit_ratio, neg_mean_distance_hits = compare_pose_lines_3(query_pose_lines, target_pose_lines)
            compare_results.append((combined_ratio, hit_ratio, neg_mean_distance_hits, target_data))
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

# indices for sorting functions
# 0: combined_ratio
# 1: hit_ratio
# 2: neg_mean_distance_hits
# 3: r_combi1
# 4: r_combi2
# 5: r_combi3
# 6: r_combi4
# 7: r_addition
# 8: target_data

def cr_desc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:,0])][::-1]
    return sorted_compare_results

def nmd_desc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:,2])][::-1]
    return sorted_compare_results

def hr_nmd_desc(compare_results):
    # hr is primary and therefore second sorting key
    # nmd is seondary and therefore second first key
    sorted_compare_results = compare_results[np.lexsort((compare_results[:,2], compare_results[:,1]))][::-1]
    return sorted_compare_results


# additional methods:
def hr_additional_desc(compare_results):
    # hr is primary and therefore second sorting key
    # nmd is seondary and therefore second first key
    sorted_compare_results = compare_results[np.lexsort((-compare_results[:,7], compare_results[:,1]))][::-1]
    return sorted_compare_results

def hr_combi3_desc(compare_results):
    # hr is primary and therefore second sorting key
    # nmd is seondary and therefore second first key
    sorted_compare_results = compare_results[np.lexsort((-compare_results[:,5], compare_results[:,1]))][::-1]
    return sorted_compare_results

def hr_combi4_desc(compare_results):
    # hr is primary and therefore second sorting key
    # nmd is seondary and therefore second first key
    sorted_compare_results = compare_results[np.lexsort((-compare_results[:,6], compare_results[:,1]))][::-1]
    return sorted_compare_results

def combi1_asc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:,3])]
    return sorted_compare_results

def combi2_asc(compare_results):
    sorted_compare_results = compare_results[np.argsort(compare_results[:,4])]
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
    norm_method = arg_obj["norm_method"]
    sort_method_name = arg_obj["sort_method_name"]
    correction_angle = arg_obj["correction_angle"]
    cone_opening_angle = arg_obj["cone_opening_angle"]
    cone_scale_factor = arg_obj["cone_scale_factor"]
    cone_base_scale_factor = arg_obj["cone_base_scale_factor"]
    filter_threshold = arg_obj["filter_threshold"]
    poseline_fallback = arg_obj["poseline_fallback"]
    bisection_fallback = arg_obj["bisection_fallback"]
    glac_fallback = arg_obj["glac_fallback"]
    additional_feature_weight = arg_obj["additional_feature_weight"]
    compare_other = arg_obj["compare_other"] if "compare_other" in arg_obj else None

    setup = compare_setupA if norm_method == 'norm_by_global_action' else compare_setupB
    if sort_method_name == 'cr_desc':
        sort_method = cr_desc
    elif sort_method_name == 'nmd_desc':
        sort_method = nmd_desc
    elif sort_method_name == 'hr_nmd_desc':
        sort_method = hr_nmd_desc
    elif sort_method_name == 'hr_additional_desc':
        sort_method = hr_additional_desc
    elif sort_method_name == 'hr_combi3_desc':
        sort_method = hr_combi3_desc
    elif sort_method_name == 'hr_combi4_desc':
        sort_method = hr_combi4_desc
    elif sort_method_name == 'combi1_asc':
        sort_method = combi1_asc
    elif sort_method_name == 'combi2_asc':
        sort_method = combi2_asc
    else:
        raise NotImplementedError("sort_method: {} not implemented".format(sort_method_name))

    config["bisection"]["correction_angle"] = correction_angle
    config["bisection"]["cone_opening_angle"] = cone_opening_angle
    config["bisection"]["cone_scale_factor"] = cone_scale_factor
    config["bisection"]["cone_base_scale_factor"] = cone_base_scale_factor
    config["compare"]["filter_threshold"] = filter_threshold

    new_datastore_values = []
    for key in datastore.keys():
        poses = datastore[key]["compoelem"]["poses"]
        datastore[key]["compoelem"]["global_action_lines"] = global_action.get_global_action_lines(poses, bisection_fallback)
        datastore[key]["compoelem"]["pose_lines"] = pose_abstraction.get_pose_lines(poses, poseline_fallback)
        new_datastore_values.append(datastore[key])

    start_time = datetime.datetime.now()
    eval_dataframe, precision_curves = setup(new_datastore_values, sort_method, norm_method, glac_fallback, compare_other, additional_feature_weight)
    norm_alias = {
        "minmax_norm_by_imgrect":"Size",
        "minmax_norm_by_bbox":"Bbox",
        "norm_by_global_action":"Glac",
        "none":"None",
    }
    filename = "final2_time{}_norm{}_{}_ca{}_co{}_cs{}_cbs{}_th{}_fbPl{}_fbBis{}_fbGa{}_other{}_aw{}.pkl".format(
        start_time.strftime("%d%m%y%H%M%S"),

        norm_alias[norm_method],
        sort_method.__name__,

        correction_angle,
        cone_opening_angle,
        cone_scale_factor,
        cone_base_scale_factor,
        filter_threshold,

        poseline_fallback,
        bisection_fallback,
        glac_fallback,

        compare_other,
        additional_feature_weight,
    )
    print("filename", filename, "p@1", eval_dataframe["p@1"]["total (mean)"])
    res_summary = {
        "experiment_name": experiment_name,
        "experiment_id": filename,
        "filename": filename,
        "datetime": start_time,
        "setup": setup.__name__,
        "eval_time_s": (datetime.datetime.now() - start_time).seconds,
        "datastore_name": datastore_name,

        "eval_dataframe": eval_dataframe,
        "precision_curves": precision_curves,
        
        "config": config,

        "norm_method": norm_method,
        "compare_method": "compare_pose_lines_3",
        "sort_method": sort_method.__name__,

        "compare_other": compare_other,

        "correction_angle": correction_angle,
        "cone_opening_angle": cone_opening_angle,
        "cone_scale_factor": cone_scale_factor,
        "filter_threshold": filter_threshold,

        "poseline_fallback":"poseline_fallback",
        "bisection_fallback":"bisection_fallback",
        "glac_fallback":"glac_fallback",
    }
    pickle.dump(res_summary, open(EVAL_RESULTS_FILE_DIR+filename, "wb"))
