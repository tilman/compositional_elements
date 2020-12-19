import itertools
from typing import cast

import numpy as np
from compositional_elements.generate.bisection import get_centroids_for_bisection, get_mapped_angle
from compositional_elements.generate.pose_direction import get_pose_directions
from compositional_elements.types import *
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
    # since combination_intersections is ordered. The combi_length of the last entry will always be the highest combi_length
    filtered_combi_length = combination_intersections[-1].cone_combination_length
    filtered_cone_intersections = [v for v in combination_intersections if v.cone_combination_length == filtered_combi_length]
    return filtered_cone_intersections

def get_global_action_lines(poses) -> Sequence[GlobalActionLine]:
    cone_intersections = get_filtered_cone_intersections(poses)
    # TODO get angle by mean of only the participating combinations, 
    # we could therefore filter the poses in the nextline with an if condition
    global_angle = np.mean([get_mapped_angle(*get_centroids_for_bisection(pose.keypoints)) for pose in poses])
    # GlobalActionLine(start, end, center, area)
    global_action_lines = [GlobalActionLine(cast(Point, v.shape.centroid), global_angle, v.shape.area, v.shape) for v in cone_intersections]
    return global_action_lines
