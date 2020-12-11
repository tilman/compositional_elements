from typing import Iterator, Tuple, cast

import cv2
import numpy as np

from compositional_elements import visualize
from compositional_elements.generate.bisection import get_bisection_point
from compositional_elements.config import config
from compositional_elements.detect.converter import k, p
from compositional_elements.types import *


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
    # type 1
    # left_bisection_keypoints: Sequence[Keypoint] = np.array(pose.keypoints)[[4,6,12]]
    # left_base_point: Keypoint = pose.keypoints[6]
    # right_bisection_keypoints: Sequence[Keypoint] = np.array(pose.keypoints)[[3,10,11]]
    # right_base_point: Keypoint = pose.keypoints[10]
    # if len(left_bisection_keypoints) == 3:
    #     left_bisection = get_bisection_point(*left_bisection_keypoints)
    # else:
    #     raise ValueError('Left pose part is missing some points for pose calculation!')
    # if len(right_bisection_keypoints) == 3:
    #     right_bisection = get_bisection_point(*right_bisection_keypoints)
    # else:
    #     raise ValueError('Right pose part is missing some points for pose calculation!')

    # left_pose_direction = PoseDirection(Point(k(left_base_point)), left_bisection)
    # right_pose_direction = PoseDirection(Point(k(right_base_point)), right_bisection)

    # type 2
    bisection_keypoint_pairs: Sequence[Tuple[Keypoint,Keypoint]] = list(zip(np.array(pose.keypoints)[[4,6,12]], np.array(pose.keypoints)[[3,10,11]]))
    if len(bisection_keypoint_pairs) != 3:
        raise ValueError('missing some points for pose calculation!')
    keypoint_pairs = [Keypoint(*p(cast(Point, LineString([k(a),k(b)]).centroid))) for a,b in bisection_keypoint_pairs]
    middle_bisection = get_bisection_point(*keypoint_pairs)
    middle_pose_direction = PoseDirection(Point(k(keypoint_pairs[1])), middle_bisection)

    # img = cv2.imread('/Users/Tilman/Documents/Programme/Python/forschungspraktikum/PoseBasedRetrievalDemo/src/API/data/intermediate_results/test.jpg')
    # img = visualize.pose(pose, img)
    # img = visualize.pose(Pose(keypoint_pairs), img, [0, 1, 2], "_m")
    # #img = visualize.pose_direction(left_pose_direction, img, (255, 0, 0)) # blue
    # #img = visualize.pose_direction(middle_pose_direction, img, (0, 255, 0)) # green
    # #img = visualize.pose_direction(right_pose_direction, img, (0, 0, 255)) # red
    # visualize.draw_window('pose_lines', img)
    return middle_pose_direction
