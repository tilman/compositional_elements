from typing import Sequence, Tuple

import cv2
import numpy as np
from numpy.core import numeric

from .config import config
from .converter import k, p
from .types import *


def create_blank() -> Sequence[Sequence[int]]:
    img = np.array([[[255,255,255]]*config["dim"][0]]*config["dim"][1], np.uint8)
    return img

def pose_lines(pose_lines: Sequence[PoseLine], img=None) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for pose_line in pose_lines:
        cv2.line(img, p(pose_line.top), p(pose_line.bottom), (0,255,0), 5)
    return img

def pose_triangles(pose_lines: Sequence[PoseTriangle], img=None) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for pose_triangle in pose_lines:
        cv2.line(img, k(pose_triangle.top), k(pose_triangle.left), (0,255,255), 5)
        cv2.line(img, k(pose_triangle.top), k(pose_triangle.right), (0,255,255), 5)
        cv2.line(img, k(pose_triangle.left), k(pose_triangle.right), (0,255,255), 5)
    return img

def poses(poses: Poses, img=None, keypoint_filter=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for p in poses:
        pose(p, img, keypoint_filter)
    return img

def pose(pose: Pose, img=None, keypoint_filter=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], suffix="") -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for i, kp in enumerate(pose.keypoints):
        if i in keypoint_filter:
            color = config["keypoint_colors"][i]
            cv2.circle(img, k(kp), 5, color, -1)
            cv2.putText(img, str(i)+suffix, k(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return img

def point_tuple_line(tuple: Tuple[Point, Point], img=None) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    cv2.line(img, p(tuple[0]), p(tuple[1]), (0,255,255), 5)
    return img

def pose_directions(pose_directions: Sequence[PoseDirection], img=None) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for dir in pose_directions:
        img = pose_direction(dir, img)
    return img

def pose_direction(dir: PoseDirection, img=None, color=(0,255,255)) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    cv2.arrowedLine(img, p(dir.start), p(dir.end), color, 5)
    return img

def pose_cluster_hull(cluster_hull: Polygon, img=None, color=(0,255,255)) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    cv2.polylines(img, [np.array(cluster_hull.exterior.coords[:-1], np.int)], False, color, 3)
    return img

def draw_window(name: str, img: Sequence[Sequence[int]]):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)

def safe(file_path: str, img: Sequence[Sequence[int]]):
    cv2.imwrite(file_path, img)
