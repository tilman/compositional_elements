from typing import Any
from compoelem.types import *
from compoelem.config import config

#optimize with KD trees or faiss: https://github.com/facebookresearch/faiss

# def euclidean_distance(a: Keypoint, b: Keypoint) -> float:
#     return 0

# def compare_global_action_line(a: GlobalActionLine], b: GlobalActionLine) -> float:
#     return 0

# def compare_global_action_lines(a: Sequence[GlobalActionLine], b: Sequence[GlobalActionLine]) -> float:
#     return 0

def compare_pose_line(a: PoseLine, b: PoseLine) -> float:
    return np.mean([a.top.distance(b.top), a.bottom.distance(b.bottom)])

def find_nearest_pose_line_distance(target: PoseLine, lines: Sequence[PoseLine]) -> Tuple[float, int]:
    # if(len(lines)):
    #     return config["compare"]["no_pose_line_penalty"]
    comparisons: Sequence[Tuple[float, int]] = []
    for idx, line in enumerate(lines):
        mean_distance = compare_pose_line(line, target)
        comparisons.append((mean_distance, idx))
    comparisons = np.array(comparisons)
    nearest_pose_line = comparisons[np.argmin(comparisons[:,0])] #type: ignore
    return nearest_pose_line

def compare_pose_lines(a: Sequence[PoseLine], b: Sequence[PoseLine]) -> float:
    if(len(b) == 0):
        return len(a) * config["compare"]["no_pose_line_penalty"] 
    nearest_pose_line_distances: Sequence[Tuple[float, int]] = []
    for query_pose_line in a:
        nearest_pose_line_distances.append(find_nearest_pose_line_distance(query_pose_line, b))
    nearest_pose_line_distances = np.array(nearest_pose_line_distances)
    nearest_pose_line_sum = np.sum(nearest_pose_line_distances[:, 0]) #type: ignore
    penalty_pose_line_indices = [idx for idx in range(0,len(b)) if idx not in nearest_pose_line_distances[:,1]] #type: ignore
    penalty_pose_lines: Sequence[PoseLine] = np.array(b)[penalty_pose_line_indices]
    if len(penalty_pose_lines) > 0:
        penalty_sum = np.sum(np.array([find_nearest_pose_line_distance(line, a) for line in penalty_pose_lines])[:,0]) 
        return nearest_pose_line_sum + penalty_sum
    else:
        return nearest_pose_line_sum

# second approach to compare the poselines. 
# Idea is to get an approaximation of the bipartite minimum graph of pose combinations weighted by distance
# => then filter out all poses above a threshold and result will be: (pose count matched)/max(pose count query img, pose count target img)
def compare_pose_lines_2(a: Sequence[PoseLine], b: Sequence[PoseLine]) -> Tuple[float, float, float]:
    if(len(b) == 0 or len(a) == 0):
        return (0, 0, 0) # since there are 0 poses to match
    pose_dist_tuple: Sequence[Tuple[float, int, int]] = [] # dist, query_idx, target_idx
    for query_idx, query_pose_line in enumerate(a):
        for target_idx, target_pose_line in enumerate(b):
            pose_dist_tuple.append((compare_pose_line(query_pose_line, target_pose_line), query_idx, target_idx))
    pose_dist_tuple_np = np.array(pose_dist_tuple)
    pose_dist_tuple_sorted = pose_dist_tuple_np[np.argsort(pose_dist_tuple_np[:,0], axis=0)]
    used_query_pose_idx = []
    used_target_pose_idx = []
    res = []
    for t in pose_dist_tuple_sorted:
        if t[1] not in used_query_pose_idx and t[2] not in used_target_pose_idx:
            res.append(t)
            used_query_pose_idx.append(t[1])
            used_target_pose_idx.append(t[2])
    res_np =  np.array(res)
    max_pose_count = max(len(a), len(b))
    threshold = 1/(max_pose_count * 2)+0.05
    res_filtered = res_np[res_np[:,0] < config["compare"]["filter_threshold"]]
    # res_filtered = res_np[res_np[:,0] < threshold] #TODO another idea: make threshold dynamic and depend on amount of poses in image => reason: more people in one image means that chances are high for a matching pose. To reduce chance => reduce the threshold
    if len(res_filtered) == 0:
        neg_mean_distance_hits = 0
    else:
        neg_mean_distance_hits = config["compare"]["filter_threshold"] - np.sum(res_filtered[:,0])/len(res_filtered)
        # guter match => neg md hoch (max: filter threshold wenn mean dist = 0)
        # schlechter match => neg md gering (0)
    hit_ratio = len(res_filtered) / max(len(a), len(b))
    # hit ratio: 
    # print(threshold, len(res_filtered), len(a), len(b), hit_ratio, neg_mean_distance_hits)
    return (hit_ratio * neg_mean_distance_hits), hit_ratio, neg_mean_distance_hits

# third is same as second but normalizes neg md between 0 and 1
def compare_pose_lines_3(a: Sequence[PoseLine], b: Sequence[PoseLine]) -> Tuple[float, float, float]:
    if(len(b) == 0 or len(a) == 0):
        return (0, 0, 0) # since there are 0 poses to match
    pose_dist_tuple: Sequence[Tuple[float, int, int]] = [] # dist, query_idx, target_idx
    for query_idx, query_pose_line in enumerate(a):
        for target_idx, target_pose_line in enumerate(b):
            pose_dist_tuple.append((compare_pose_line(query_pose_line, target_pose_line), query_idx, target_idx))
    pose_dist_tuple_np = np.array(pose_dist_tuple)
    pose_dist_tuple_sorted = pose_dist_tuple_np[np.argsort(pose_dist_tuple_np[:,0], axis=0)]
    used_query_pose_idx = []
    used_target_pose_idx = []
    res = []
    for t in pose_dist_tuple_sorted:
        if t[1] not in used_query_pose_idx and t[2] not in used_target_pose_idx:
            res.append(t)
            used_query_pose_idx.append(t[1])
            used_target_pose_idx.append(t[2])
    res_np =  np.array(res)
    # max_pose_count = max(len(a), len(b))
    # threshold = 1/(max_pose_count * 2)+0.05
    res_filtered = res_np[res_np[:,0] < config["compare"]["filter_threshold"]]
    # res_filtered = res_np[res_np[:,0] < threshold] #TODO another idea: make threshold dynamic and depend on amount of poses in image => reason: more people in one image means that chances are high for a matching pose. To reduce chance => reduce the threshold
    if len(res_filtered) == 0:
        neg_mean_distance_hits = 0
    else:
        neg_mean_distance_hits = (config["compare"]["filter_threshold"] - np.sum(res_filtered[:,0])/len(res_filtered)) / config["compare"]["filter_threshold"]
        # guter match => neg md hoch (max = 1)
        # schlechter match => neg md gering (min = 0)
    hit_ratio = len(res_filtered) / max(len(a), len(b))
    # guter match => neg hr hoch (max = 1)
    # schlechter match => neg hr gering (min = 0)
    # guter match => neg cr hoch (max = 1)
    # schlechter match => neg cr gering (min = 0)
    return (hit_ratio * neg_mean_distance_hits), hit_ratio, neg_mean_distance_hits

# if we normalize by action center we are getting more than one normalization results because we can have multiple action centers
# because we only want one similarity score for each pair of pictures we then filter for the lowest similarity between all combination of normalizations
def filter_pose_line_ga_result(ga_res: Sequence[Tuple[float, float, float, Any]]) -> Tuple[float, float, float, Any]:
    # combined_ratio, hit_ratio, neg_mean_distance_hits, target_data
    np_ga_res = np.array(ga_res)
    ga_res_filtered_by_hit_ratio = np_ga_res[np_ga_res[:,1] == max(np.array(ga_res)[:,1])] #first filter elem with max hit ratio
    ga_res_filtered_by_mean_dist = ga_res_filtered_by_hit_ratio[ga_res_filtered_by_hit_ratio[:,2] == max(np.array(ga_res_filtered_by_hit_ratio)[:,2])] #then filter elem with max neq md
    return ga_res_filtered_by_mean_dist[0]