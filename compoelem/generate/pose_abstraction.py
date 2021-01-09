from typing import Tuple
import numpy as np

from compoelem.config import config
from compoelem.types import *

def get_pose_lines(poses: Poses) -> Sequence[PoseLine]:
    pose_lines: Sequence[PoseLine] = [];
    for pose in poses:
        try:
            pose_abstraction = get_pose_abstraction(pose)
            pose_lines.append(pose_abstraction)
        except ValueError as e:
            print(e)
    return pose_lines

def get_pose_triangles(poses: Poses) -> Sequence[PoseTriangle]:
    pose_triangles: Sequence[PoseTriangle] = [];
    for pose in poses:
        try:
            triangle = get_pose_triangle(pose)
            pose_triangles.append(triangle)
        except ValueError as e:
            print(e)
    return pose_triangles

def get_pose_abstraction(pose: Pose) -> PoseLine:
    triangle = get_pose_triangle(pose)
    poseline = get_pose_line(triangle)
    return poseline

def get_pose_triangle(pose: Pose) -> PoseTriangle:
    pose_keypoints = np.array(pose.keypoints, dtype=Keypoint)
    
    # Partition the keypoint list in three sequences corresponding to left, right, top triangle corner points
    left_keypoint_selection: Sequence[Keypoint] = pose_keypoints[config["pose_abstraction"]["keypoint_list"]["left"]].tolist()
    right_keypoint_selection: Sequence[Keypoint] = pose_keypoints[config["pose_abstraction"]["keypoint_list"]["right"]].tolist()
    top_keypoint_selection: Sequence[Keypoint] = pose_keypoints[config["pose_abstraction"]["keypoint_list"]["top"]].tolist()

    # Select first keypoint of each partition witch is not None
    left_keypoints = list(filter(lambda kp: not kp.isNone, left_keypoint_selection))
    right_keypoints = list(filter(lambda kp: not kp.isNone, right_keypoint_selection))
    top_keypoints = list(filter(lambda kp: not kp.isNone, top_keypoint_selection))
    if(len(left_keypoints) == 0):
        raise ValueError('missing valid left keypoints for triangle calculation!')
    if(len(right_keypoints) == 0):
        raise ValueError('missing valid left keypoints for triangle calculation!')
    if(len(top_keypoints) == 0):
        raise ValueError('missing valid left keypoints for triangle calculation!')

    return PoseTriangle(top_keypoints[0], right_keypoints[0], left_keypoints[0])

def get_pose_line(triangle: PoseTriangle) -> PoseLine:
    bottom_line = LineString([[triangle.left.x, triangle.left.y], [triangle.right.x, triangle.right.y]])
    top_point = Point(triangle.top.x, triangle.top.y)
    bottom_point = Point(bottom_line.centroid)
    return PoseLine(top_point, bottom_point)