from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np

from shapely.geometry import LineString, Point, Polygon

Img = Sequence[Sequence[int]]

class Keypoint:
    x: int
    y: int
    score: float
    isNone: bool
    def __init__(self, x: int, y: int, score: float = 0):
        self.x = int(x)
        self.y = int(y)
        self.score = score
        self.isNone = False
    def __str__(self) -> str:
        return "Kp ({},{})".format(self.x, self.y)
    def __repr__(self) -> str:
        return "Kp ({},{})".format(self.x, self.y)

class NoneKeypoint(Keypoint):
    x: int
    y: int
    isNone: bool
    score: int
    def __init__(self):
        self.x = -1
        self.y = -1
        self.score = 0
        self.isNone = True
    def __str__(self) -> str:
        return "N Kp"
    def __repr__(self) -> str:
        return "N Kp"

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
    def __init__(self, center: Point, angle: float, area: float, intersection_shape: Polygon = None):
        self.angle = angle
        self.area = area
        self.center = center
        if intersection_shape:
            self.intersection_shape = intersection_shape
        else:
            self.intersection_shape = Polygon()

        dist = 2000 # FIXME: move hardcoded value to config, or maybe calculate dist based on area
        if np.isnan(angle):
            x_offset = 0
            y_offset = 0
            print("Warning: GlobalActionLine angle is NaN")
        else:
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
