import itertools
from typing import cast

import numpy as np
from compoelem.generate.bisection import get_centroids_for_bisection, get_angle_in_respect_to_x
from compoelem.generate.pose_direction import get_pose_directions
from compoelem.types import *
from shapely.geometry.polygon import Polygon


def get_cone_combination_intersections(pose_directions: Sequence[PoseDirection]) -> Sequence[ConeIntersection]:
    """calculate every intersection of all combinations of input pose list.
    It will return all combination intersections which are not null.
    So we can later for example select the intersection with the highest combination_length as our target.

    Args:
        pose_directions (Sequence[PoseDirection]): Output of get_pose_directions(poses)

    Returns:
        Sequence[Tuple[Combination, int, Polygon]]: a ordered list of pairs (combination_length, combination, combi_intersection_result)
                                                    ordered with ascending combination_length
    """
    combination_intersections: Sequence[ConeIntersection] = []
    # we increase the length of the combinations of cones. Ending in a combination_length with all cones in one combination
    # len(pose_directions) == 7:
    #   combination_length = 1 => [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]
    #   combination_length = 2 => [(0, 1), ..., (0, 7), (1, 2), ..., (1, 7), (2, 3), ...]   # no duplicate combinations live (1, 0) since intersection would be the same
    #   combination_length = 8 => [(0, 1, 2, 3, 4, 5, 6, 7)]
    for combination_length in range(1,len(pose_directions)+1):
        cone_combinations: Sequence[ConeCombination] = list(itertools.combinations(range(0,len(pose_directions)), combination_length))
        # generating all combinations of cones with combi length r
        for combination in cone_combinations:
            # start the recursive intersection calculation with the first entry of the combination tuple as the target cone
            combi_intersection_result = pose_directions[combination[0]].cone
            for i in combination[1:]:
                # iterate over the remaining combinations and recursivly call the intersection on the current intersection
                # this will slowly make the intersection smaller till all intersections from the combination are generated
                combi_intersection_result = cast(Polygon, combi_intersection_result.intersection(pose_directions[i].cone))
            if not combi_intersection_result.is_empty:
                # only append the final intersection if the target cone is not empty
                # also store the combination tuple
                combination_intersections.append(ConeIntersection(combi_intersection_result, combination))
    return combination_intersections

def get_filtered_cone_intersections(poses) -> Sequence[ConeIntersection]:
    """get the filtered cone intersections from the poses. We filter by selecting the intersections with the most cones participating. Therfore it could happen that the amount of intersections is greater than 1.

    Args:
        poses ([type]): converted HRNet output

    Returns:
        Sequence[Polygon]: Each entry is a cone intersection represented by a Polygon
    """
    pose_directions = get_pose_directions(poses)
    combination_intersections = get_cone_combination_intersections(pose_directions)
    if len(combination_intersections) == 0:
        return []
    # since combination_intersections is ordered. The combi_length of the last entry will always be the highest combi_length
    filtered_combi_length = combination_intersections[-1].cone_combination_length
    filtered_cone_intersections = [v for v in combination_intersections if v.cone_combination_length == filtered_combi_length]
    return filtered_cone_intersections

def get_combined_angle(poses) -> float:
    """calculates and combines the bisection angle of all input poses.

    Args:
        poses ([type]): Openpose transformed output poses

    Returns:
        float: angle in radians
    """
    angles: Sequence[float] = []
    for pose in poses:
        try:
            angles.append(get_angle_in_respect_to_x(*get_centroids_for_bisection(pose.keypoints)))
        except ValueError as e:
            #print(e)
            pass
    return np.mean(angles) * -1

def get_global_action_lines(poses) -> Sequence[GlobalActionLine]:
    """calculate global action lines. Therefore we calculate the intersecting pose direction cones and take the centroid of
    the intersection with the most cones participating. From these participating cones we also calculate the angle from the
    bisection vector and average it together to calculate the angle for the global action line. The line is then drawn trough
    the intersection area centroid with this newly calculated centroid.

    Args:
        poses ([type]): Openpose transformed output poses

    Returns:
        Sequence[GlobalActionLine]: a single or multiple global action lines. Mostly a single line but if there are multiple 
        different intersection areas with the same amount of cones participating we take all of them.
    """
    cone_intersections = get_filtered_cone_intersections(poses)
    global_action_lines = []
    for cone_intersection in cone_intersections:
        filtered_participating_poses = np.array(poses)[np.array(cone_intersection.cone_combination)]
        combined_angle = get_combined_angle(filtered_participating_poses)
        global_action_lines.append(
            GlobalActionLine(
                cast(Point, cone_intersection.shape.centroid),
                combined_angle,
                cone_intersection.shape.area,
                cone_intersection.shape
            )
        )
    return global_action_lines
