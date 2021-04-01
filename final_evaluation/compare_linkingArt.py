# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import numpy as np
import cv2
import pandas as pd
import datetime
import torch
from tqdm import tqdm
from tqdm.std import trange
from . import eval_utils
from itertools import combinations
from compoelem.detect.openpose.lib.utils.common import BodyPart, Human, CocoPart

def neg_cos_dist(r_tick, s_tick):
    a = r_tick.flatten()
    b = s_tick.flatten()
    return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) # type: ignore, manually checked it works!

def flipped_cosine_min_dist(r_tick, s_tick):
    s_star = np.array([[-kp[0], kp[1]] for kp in s_tick])
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

def neck_norm_poses(r, s):
    ROOT_POINT = CocoPart.Neck.value
    r_root = r[ROOT_POINT]
    s_root = s[ROOT_POINT]
    RSHOULDER = CocoPart.RShoulder.value
    LSHOULDER = CocoPart.LShoulder.value
    if(isNoneKp(r_root)): # extension to the paper: if neck point is missing we try to esitmate it with the midpoint of left and right shoulder
        if(isNoneKp(r[RSHOULDER]) or isNoneKp(r[LSHOULDER])):
            raise ValueError("neck point and shoulder point missing, normalization not possible, skipping that pose")
        else:
            r_root = [(r[RSHOULDER][0]+r[LSHOULDER][0])/2, (r[RSHOULDER][1]+r[LSHOULDER][1])/2]
    if(isNoneKp(s_root)): # extension to the paper: if neck point is missing we try to esitmate it with the midpoint of left and right shoulder
        if(isNoneKp(s[RSHOULDER]) or isNoneKp(s[LSHOULDER])):
            raise ValueError("neck point and shoulder point missing, normalization not possible, skipping that pose")
        else:
            r_root = [(s[RSHOULDER][0]+s[LSHOULDER][0])/2, (s[RSHOULDER][1]+s[LSHOULDER][1])/2]
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
        return (2, []) #maximum possible neg cos dist
    else:
        # return min(dist)
        am = np.argmin(np.array(dist))
        return (dist[am], combinations[am])

def compare_dist_bipart(poses_i1, poses_i2): #in paper this is dist_t(i1,i2)
    t = 0.05
    poses_i1 = np.array([openpose_to_nparray(human) for human in poses_i1]) # output shape of each item is (18, 2) since we are using the 18 openpose keypoint model
    poses_i2 = np.array([openpose_to_nparray(human) for human in poses_i2])
    all_dist = []
    all_combinations = []
    for idx_r, r in enumerate(poses_i1):
        dist = []
        combinations = []
        for idx_s, s in enumerate(poses_i2):
            try:
                r_tick, s_tick = neck_norm_poses(r, s)
            except ValueError as e: # "neck point missing, normalization not possible, skipping that pose"  => this edge case is not mentioned in the paper but was the only sensible decision I think
                #print(e) 
                continue
            dist.append(flipped_cosine_min_dist(r_tick, s_tick))
            combinations.append((idx_r, idx_s))
        if len(dist) == 0:
            all_dist.append(t) #dist can be empty if r has no neck point. => return t as there is no pose matching
        else:
            am = np.argmin(np.array(dist))
            if dist[am] <= t:
                all_dist.append(dist[am])
                all_combinations.append(combinations[am])
            else:
                all_dist.append(t)
    dist_sum = np.sum(all_dist)
    return (dist_sum, all_combinations)

def verify_inliers(r,s,transformation):
    inlier_threshold = calc_ransac_inlier_threshold(r,s)
    # apply transformation on all keypoints of s (in paper: projection to the query image)
    # then: A pair of keypoints is considered consistent with a transformation when the keypoint from the potential image match is within a specified distance from the query image keypoint.
    # This threshold distance is relative with respect to the estimated query image pose size and is therefore different for each pose in the query image.
    s_transformed = np.array([([*si,1] @ transformation)[0:2] for si in s])
    inlier_mask = [np.linalg.norm(ri-si) < inlier_threshold and (ri[0] != 0 or ri[1] != 0) and (si[0] != 0 or si[1] != 0) for ri, si in zip(r, s_transformed)] #with if: only check points where both points calculated by openpose
    # return => indices of consistent keypoints
    return np.array(range(0, len(r)))[inlier_mask]

def calc_ransac_inlier_threshold(r,s):
    # To determine the inlier threshold for RANSAC, relative query pose size with respect to the canonical pose size is estimated.
    # For a canonical pose, the distances between connected pose keypoints are known. 
    # The relative query pose size is computed as a median of the ratios between distances of connected keypoints detected in the query pose and corresponding distances in the canonical pose.
        # from above:( This threshold distance is relative with respect to the estimated query image pose size and is therefore different for each pose in the query image.)

    # ???? => => What is the canonical pose?
    return 0.01 # TODO implement function, this fixed value is just for testing

def calc_geometric_transformation(r, s): #maybe also two kp from each
    # INFO r/s has to be of shape (2, 2) for initial calc and (amount of inliers, 2) for reestimate
    # The transformation consists of scale, translation and horizontal flip.  => return transformation matrix here???
    # Using two keypoint correspondences (two keypoint pairs?), the transformation is estimated in terms of least-squares. 
    #   => An exact solution to the system of equations does not exist as the transformation 
    #      has three degrees of freedom and there are four equations resulting in an overdetermined system.

    # ???? Really not sure about this part here. I think this is what we need for the least square fit but not sure about it.
    # How to limit the affine transformation to only scale and translation?
    # A should be of the shape:
    # [[a, 0, e]]
    # [[0, b, f]]
    # [[0, 0, 1]]
    # with a and b defining x and y scale. And e and f defining x and y translation.
    R = np.hstack([r, np.ones((r.shape[0], 1))])
    S = np.hstack([s, np.ones((s.shape[0], 1))])
    A, residuals, rank, singular_values = np.linalg.lstsq(S, R, rcond=None) # we want to transform s points to query image. So perform `s @ A => r'`
    print(A)
    # zero out these values to get only scale and translation
    # print("A",A)
    # A[0,1] = 0
    # A[1,0] = 0
    # A[2,0] = 0
    # A[2,1] = 0
    # A[2,2] = 1
    # print("A",A)
    return A, sum(residuals) # seems like residuals is always an empty array => check np.linalg.lstsq again

# def reestimate_geometric_transformation_least_square(r_inliers, s_inliers): => is the same as calc_geometric_transformation
#     # Once a transformation with a sufficient number of inliers is found, all keypoint correspondences consistent with it are used to re-estimate the transformation in terms of least squares
#    pass

def estimate_geometric_transformation_ransac(r,s):
    s_star = np.array([[-kp[0], kp[1]] for kp in s]) # s_star is flipped pose

    combi = np.array(list(combinations(range(0,18), r=2)))
    for idx1, idx2 in combi[np.random.choice(len(combi), len(combi), replace=False)]:  # 1) sample 2 keypoint indices -> idx1 and idx2, and then take the keypoints from both images:
        # 2) transformation          =    calc_geometric_transformation(r[idx1], r[idx2], s[idx1], s[idx2]) or (kp1, kp2)
        #    transformation_flipped  =    calc_geometric_transformation(kpr1, kpr2, kps_star1, kps_star2) or (kp1, kp2)
        two_kp_sample = [idx1,idx2]
        transformation_normal, res_sum_error_normal = calc_geometric_transformation(r[two_kp_sample], s[two_kp_sample])
        transformation_flipped, res_sum_error_flipped = calc_geometric_transformation(r[two_kp_sample], s_star[two_kp_sample])
        # the transformation with a smaller error on the two keypoint correspondences is chosen. => what error => residual sum of least squares fit
        if res_sum_error_normal <= res_sum_error_flipped:
            transformation = transformation_normal
            s_used = s
        else:
            transformation = transformation_flipped
            s_used = s_star
        # print("ransac lrs",len(r),len(s))
        inliers = verify_inliers(r,s_used,transformation)
        print("sample_kp", two_kp_sample, "inliers_kp", inliers)
        #break as soon as 18/4 => 5 inliers are found and then return!!!
        #print("inliers",len(inliers))
        if(len(inliers) >= 5 ): #break as soon as 18/4 => 5 inliers are found and then return!!!
            # The output of the RANSAC method is the best transformation found, measured by the number of inliers
            return (transformation, inliers) # return: transformation + inliers: The transformation is found with RANSAC [27], so that it has the largest number of inliers, i.e. keypoints consistent with the transformation
    return (None, None)

def robust_verify(poses_i1, poses_i2, neck_norm):  #TODO: Test out both. normed to neckpoint and normal. Not sure if we are working with the right input data => normalised around root or not?? Keypoints normed to imgrect between 0 and 1 or with pixel values?
    poses_i1 = np.array([openpose_to_nparray(human) for human in poses_i1]) # output shape of each item is (18, 2) since we are using the 18 openpose keypoint model
    poses_i2 = np.array([openpose_to_nparray(human) for human in poses_i2])
    transformations = [] # list of refined transformations
    validated_pairs = []
    unvalidated_pairs = []
    for idx_r, r in enumerate(poses_i1):  # For each tentative pose correspondence (one figure in the query image, one figure in the database image), a geometric transformation is estimated.
        for idx_s, s in enumerate(poses_i2):
            if neck_norm:
                try:
                    r, s = neck_norm_poses(r, s)
                except ValueError:
                    continue
            transformation, inliers = estimate_geometric_transformation_ransac(r,s)
            # If the number of inliers is sufficient, the corresponding pose pair is considered validated, otherwise, the transformation is filtered out.
            # Once a transformation with a sufficient number of inliers is found, all keypoint correspondences consistent with it are used to re-estimate the transformation in terms of least squares
            # If the number of inliers is sufficient, the corresponding pose pair is considered validated, otherwise, the transformation is filtered out.
            # 18 keypoints / 4 => 4,5 => 5 # The poses are considered validated if the estimated transformation aligned at least 1/4 of all keypoints of the pose, which is 7 out of 25 in our setup.
            if inliers is not None:
                final_transformation, sum_residuals = calc_geometric_transformation(r[inliers], s[inliers]) #save for later => Each transformation, corresponding to a different pair of poses, is applied on all validated pairs of poses, transforming other figures in the image
                validated_pairs.append((idx_r, idx_s))
                transformations.append(final_transformation)
            else:
                unvalidated_pairs.append((idx_r, idx_s))


    # Each transformation, corresponding to a different pair of poses, is applied on all validated pairs of poses, transforming other figures in the image.
    #  The maximum number of keypoints consistent with a transformation is used as a measure of image similarity.
    total_consistent = []
    for transformation in transformations:
        consistent_for_this_transformation = []
        for idx_r, idx_s in validated_pairs:
            r = poses_i1[idx_r]
            s = poses_i2[idx_s]
            # print("rob.ver. lrs",len(r),len(s))
            vi = verify_inliers(r, s, transformation)
            if vi is not None:
                consistent_for_this_transformation.append(len(vi))
        total_consistent.append(sum(consistent_for_this_transformation))
    return 0 if len(total_consistent) == 0 else max(total_consistent)
    
def compare(data, sort_method, compare_method, neck_norm):
    res_metrics = {}
    for query_data in tqdm(data, total=len(data)):
        compare_results = []
        for target_data in data:
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            distance, combinations = compare_method(query_data["compoelem"]["humans"], target_data["compoelem"]["humans"])
            compare_results.append((distance, combinations, target_data)) # we will use the same precomputed poses as we already computed for compoelem method
        compare_results = np.array(compare_results)
        sorted_compare_results = sort_method(compare_results)

        l = 50
        # cut sorted_compare_results into first segment [0:l] and remaining segment [l:-1]
        # for first segment perform RANSAC (robust verification) and set matched/unmatched
        # for last segment set all to unmatched
        compare_results = [(0, *r) for r in sorted_compare_results[l:-1]] #TODO check if padding with 0 is really what we want. Since idx0 stand for max(total_consistent) in the robust_verify result i guess so
        for target in sorted_compare_results[0:l]:
            # => TODO: pose pairs whose distance exceeds 0.1 are discarded => means attached to list without robust_verify
            target_data = target[-1]
            if query_data["className"] == target_data["className"] and query_data["imgName"] == target_data["imgName"]:
                continue
            match_count = robust_verify(query_data["compoelem"]["humans"], target_data["compoelem"]["humans"], neck_norm)
            compare_results.append((match_count, *target))
        compare_results = np.array(compare_results)
        # then perform lexsort, Sort in a manner that higher match count comes to the front and unmatched to the back. Second criteria is than distance from above
        sorted_compare_results = compare_results[np.lexsort((compare_results[:,1], -compare_results[:,0]))] # first level of sorting is 0 (robust verification), and then 1 (distance)


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
    for neck_norm in [True, False]:
        for compare_method in [compare_dist_bipart, compare_dist_min]:
            start_time = datetime.datetime.now()
            sortmethod = sort_asc
            experiment_id = "datastore: {}, compare_method: {}, sort_method: {}, neck_norm:{}".format(datastore_name, compare_method.__name__, sortmethod.__name__,neck_norm)
            print("EXPERIMENT:",experiment_id)
            eval_dataframe = compare(list(datastore.values()), sortmethod, compare_method, neck_norm)
            all_res_metrics.append({
                "experiment_id": experiment_id,
                "datetime": start_time,
                "eval_time_s": (datetime.datetime.now() - start_time).seconds,
                "datastore_name": datastore_name,
                "neck_norm": neck_norm,
                "compare_method": compare_method.__name__,
                "sort_method": sortmethod.__name__,
                "eval_dataframe": eval_dataframe,
                "linkingArt":True,
                "new":True,
            })
    return all_res_metrics