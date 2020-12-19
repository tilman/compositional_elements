from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np

from shapely.geometry import LineString, Point, Polygon

class Keypoint:
    x: int
    y: int
    isNone: bool
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.isNone = False

class NoneKeypoint(Keypoint):
    x: int
    y: int
    isNone: bool
    def __init__(self):
        self.x = -1
        self.y = -1
        self.isNone = True

@dataclass
class Pose:
    keypoints: Sequence[Keypoint]

Poses = Sequence[Pose]

class PoseTriangle:
    shape: Polygon
    top: Keypoint
    left: Keypoint
    right: Keypoint
    def __init__(self, top: Keypoint, left: Keypoint, right: Keypoint): 
        self.top = top
        self.left = left
        self.right = right
        self.shape = Polygon([[top.x, top.y], [left.x, left.y], [right.x, right.y]])

class PoseLine:
    line: LineString
    top: Point
    bottom: Point
    def __init__(self, top:Point, bottom: Point): 
        self.top = top
        self.bottom = bottom
        self.line = LineString([[top.x, top.y], [bottom.x, bottom.y]])

ConeCombination = Tuple[int, ...]

class ConeIntersection:
    shape: Polygon
    cone_combination: ConeCombination
    cone_combination_length: int
    def __init__(self, shape: Polygon, cone_combination: ConeCombination): 
        self.shape = shape
        self.cone_combination = cone_combination
        self.cone_combination_length = len(cone_combination)

class GlobalActionLine:
    line: LineString
    start: Point
    end: Point
    center: Point
    area: float
    angle: float
    intersection_shape: Polygon
    def __init__(self, center: Point, angle:float, area: float, intersection_shape: Polygon = None):
        self.angle = angle
        self.area = area
        self.center = center
        if intersection_shape:
            self.intersection_shape = intersection_shape
        else:
            self.intersection_shape = Polygon()

        dist = 2000 # FIXME: move hardcoded value to config, or maybe calculate dist based on area
        x_offset = int(dist * np.cos(angle))
        y_offset = int(dist * np.sin(angle))
        self.start = Point(center.x - x_offset, center.y - y_offset)
        self.end = Point(center.x + x_offset, center.y + y_offset)


class PoseDirection:
    line: LineString
    start: Point
    end: Point
    cone: Polygon
    def __init__(self, start:Point, end: Point, cone: Polygon): 
        self.start = start
        self.end = end
        self.line = LineString([[start.x, start.y], [end.x, end.y]])
        self.cone = cone
