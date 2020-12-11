import numpy as np

from compositional_elements.config import config
from compositional_elements.types import *


def get_poselines(poses: Poses) -> Sequence[PoseLine]:
    pose_lines: Sequence[PoseLine] = [];
    for pose in poses:
        pose_lines.append(get_pose_abstraction(pose))
    return pose_lines

def get_posetriangles(poses: Poses) -> Sequence[PoseTriangle]:
    pose_triangles: Sequence[PoseTriangle] = [];
    for pose in poses:
        pose_triangles.append(get_posetriangle(pose))
    return pose_triangles

def get_pose_abstraction(pose: Pose) -> PoseLine:
    triangle = get_posetriangle(pose)
    poseline = get_poseline(triangle)
    return poseline

def get_posetriangle(pose: Pose) -> PoseTriangle:
    pose_keypoints = np.array(pose.keypoints, dtype=Keypoint)
    
    # Partition the keypoint list in three sequences corresponding to left, right, top triangle corner points
    left_keypoint_selection: Sequence[Keypoint] = pose_keypoints[config["pose_abstraction"]["keypoint_list"]["left"]].tolist()
    right_keypoint_selection: Sequence[Keypoint] = pose_keypoints[config["pose_abstraction"]["keypoint_list"]["right"]].tolist()
    top_keypoint_selection: Sequence[Keypoint] = pose_keypoints[config["pose_abstraction"]["keypoint_list"]["top"]].tolist()

    # Select first keypoint of each partition witch is not None
    left_keypoint = list(filter(lambda kp: not kp.isNone, left_keypoint_selection))[0]
    right_keypoint = list(filter(lambda kp: not kp.isNone, right_keypoint_selection))[0]
    top_keypoint = list(filter(lambda kp: not kp.isNone, top_keypoint_selection))[0]

    return PoseTriangle(top_keypoint, right_keypoint, left_keypoint)

def get_poseline(triangle: PoseTriangle) -> PoseLine:
    bottom_line = LineString([[triangle.left.x, triangle.left.y], [triangle.right.x, triangle.right.y]])
    top_point = Point(triangle.top.x, triangle.top.y)
    bottom_point = Point(bottom_line.centroid)
    return PoseLine(top_point, bottom_point)
