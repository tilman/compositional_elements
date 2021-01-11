from typing import Iterator, Tuple, cast

import cv2
import numpy as np

from compoelem import visualize
from compoelem.generate.bisection import get_angle, get_bisection_cone, get_bisection_point, get_centroids_for_bisection, keypoint_to_point
from compoelem.config import config
from compoelem.detect.converter import k, p
from compoelem.types import *


def get_pose_directions(poses: Poses) -> Sequence[PoseDirection]:
    """Generate pose directions from multiple input Poses

    Args:
        poses (Poses): Poses obtained by keypoint detection method

    Returns:
        Sequence[PoseDirection]: A list of pose directions
    """
    pose_directions: Sequence[PoseDirection] = []
    # pose_directions: Sequence[Tuple[Point, Point]] = [];
    for pose in poses:
        try:
            pose_directions.append(get_pose_direction(pose))
        except ValueError:
            print('skipping pose because of missing points')
    return pose_directions

def get_pose_direction(pose: Pose) -> PoseDirection:
    top_kp, middle_kp, bottom_kp = get_centroids_for_bisection(pose.keypoints)
    bisection_point = get_bisection_point(top_kp, middle_kp, bottom_kp)
    bisection_cone = get_bisection_cone(top_kp, middle_kp, bottom_kp)
    middle_pose_direction = PoseDirection(keypoint_to_point(middle_kp), keypoint_to_point(bisection_point), bisection_cone)
    return middle_pose_direction
