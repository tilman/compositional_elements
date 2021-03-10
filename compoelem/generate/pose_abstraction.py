from ctypes import ArgumentError
from typing import Tuple
import numpy as np

from compoelem.config import config
from compoelem.types import *
from compoelem.detect.converter import k, p

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
    try:
        triangle = get_pose_triangle(pose)
        poseline = get_pose_line(triangle)
        return poseline
    except ValueError:
        #return PoseLine(top_point, bottom_point)
        return get_fallback_pose_line(pose)

def get_fallback_pose_line(pose: Pose) -> PoseLine:
    pose_keypoints = np.array(pose.keypoints, dtype=Keypoint)
    if(pose_keypoints[0].isNone or pose_keypoints[1].isNone):
        raise AssertionError('missing valid keypoint 0 (nose) or 1 (neck) for pose line fallback!')
    
    top_keypoint_selection: Sequence[Keypoint] = pose_keypoints[config["pose_abstraction"]["keypoint_list"]["top"]].tolist()
    top_keypoints = list(filter(lambda kp: not kp.isNone, top_keypoint_selection))
    top_point = Point(*k(top_keypoints[0]))
    

    nose_point = np.array(k(pose_keypoints[0]))
    neck_point = np.array(k(pose_keypoints[1]))
    fallback_length = np.linalg.norm(nose_point-neck_point) * 2

    bottom_point = Point(neck_point[0], (neck_point[1] - fallback_length))
    return PoseLine(top_point, bottom_point)


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
    if(len(top_keypoints) == 0):
        raise AssertionError('missing valid top keypoints for triangle calculation!')
    if(len(left_keypoints) == 0):
        raise ValueError('missing valid left keypoints for triangle calculation!')
    if(len(right_keypoints) == 0):
        raise ValueError('missing valid right keypoints for triangle calculation!')

    return PoseTriangle(top_keypoints[0], right_keypoints[0], left_keypoints[0])

def get_pose_line(triangle: PoseTriangle) -> PoseLine:
    bottom_line = LineString([[triangle.left.x, triangle.left.y], [triangle.right.x, triangle.right.y]])
    top_point = Point(triangle.top.x, triangle.top.y)
    bottom_point = Point(bottom_line.centroid)
    return PoseLine(top_point, bottom_point)