from typing import Iterator, Tuple, cast

import cv2
import numpy as np
from shapely.geometry import Polygon

from compositional_elements import visualize
from compositional_elements.generate.bisection import get_bisection_point
from compositional_elements.config import config
from compositional_elements.detect.converter import k, p
from compositional_elements.types import *


def get_pose_cluster_convex_hull(poses: Poses) -> Polygon:
    flat_keypoint_list: Sequence[Tuple[int, int]] = []
    for pose in poses:
        for kp in pose.keypoints:
            if not kp.isNone:
                flat_keypoint_list.append(k(kp))
    keypoint_cloud_polygon = Polygon(flat_keypoint_list)
    return cast(Polygon, keypoint_cloud_polygon.convex_hull)


def get_pose_cluster_border_hull(poses: Poses, x_max, y_max) -> Polygon:
    flat_keypoint_list: Sequence[Tuple[int, int]] = []
    for pose in poses:
        for kp in pose.keypoints:
            if not kp.isNone:
                flat_keypoint_list.append(k(kp))
    flat_keypoint_array = np.array(flat_keypoint_list)

    left_side_keypoints: Sequence[Tuple[int, int]] = []
    right_side_keypoints: Sequence[Tuple[int, int]] = []
    for i in range(0, y_max):
        y_keypoint_line = flat_keypoint_array[flat_keypoint_array[:,1] == i][:,0]
        if len(y_keypoint_line) > 0:
            right_side_keypoints.append((max(y_keypoint_line), i))
            left_side_keypoints.append((min(y_keypoint_line), i))

    top_side_keypoints: Sequence[Tuple[int, int]] = []
    # TODO: add another loop for the top line calculation
    
    # TODO: develop a method to merge left, top and right side's together

    return Polygon(left_side_keypoints) # INFO only return left side for testing purposes
