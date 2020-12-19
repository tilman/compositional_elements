from typing import Sequence, Tuple, cast

import cv2
import numpy as np

from compositional_elements.config import config
from compositional_elements.detect.converter import k, p
from compositional_elements.types import *


def create_blank() -> Sequence[Sequence[int]]:
    img = np.array([[[255,255,255]]*config["dim"][0]]*config["dim"][1], np.uint8)
    return img

def pose_lines(pose_lines: Sequence[PoseLine], img=None) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for pose_line in pose_lines:
        cv2.line(img, p(pose_line.top), p(pose_line.bottom), (0,255,0), 5)
    return img

def global_action_lines(global_action_lines: Sequence[GlobalActionLine], img=None) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for ga_line in global_action_lines:
        if not ga_line.intersection_shape == None:
            visible_area = Polygon([(0,0),(len(img[0]), 0), (len(img[0]),len(img)), (0,len(img))])
            visible_intersection = cast(Polygon, visible_area.intersection(ga_line.intersection_shape))
            cv2.drawContours(img, [np.array(visible_intersection.exterior.coords[:], np.int)], 0, (150,100,100), -1)
            # cv2.drawContours(img, [np.array(ga_line.intersection_shape.exterior.coords[:], np.int)], 0, (150,100,100), -1)
        cv2.line(img, p(ga_line.start), p(ga_line.end), (0,255,255), 5)
        cv2.circle(img, p(ga_line.center), 10, (255,255,0), -1)
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
    for ip, p in enumerate(poses):
        pose(p, img, keypoint_filter, "_"+str(ip))
    return img

def pose(pose: Pose, img=None, keypoint_filter=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], suffix="") -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for i, kp in enumerate(pose.keypoints):
        if i in keypoint_filter:
            color = config["keypoint_colors"][i]
            cv2.circle(img, k(kp), 3, color, -1)
            cv2.putText(img, str(i)+suffix, k(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
    return img

def point_tuple_line(tuple: Tuple[Point, Point], img=None) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    cv2.line(img, p(tuple[0]), p(tuple[1]), (0,255,255), 5)
    return img

def pose_directions(pose_directions: Sequence[PoseDirection], img=None, color=(0,255,255), plotShape=False) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    for dir in pose_directions:
        img = pose_direction(dir, img, color, plotShape)
    return img

def pose_direction(dir: PoseDirection, img=None, color=(0,255,255), plotShape=False) -> Sequence[Sequence[int]]:
    if img is None:
        img = create_blank()
    cv2.arrowedLine(img, p(dir.start), p(dir.end), color, 5)
    if plotShape:
        cv2.polylines(img, [np.array(dir.cone.exterior.coords[:-1], np.int)], True, (255,0,255), 2)
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
