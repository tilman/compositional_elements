from dataclasses import dataclass
from typing import Sequence

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

class PoseDirection:
    line: LineString
    start: Point
    end: Point
    def __init__(self, start:Point, end: Point): 
        self.start = start
        self.end = end
        self.line = LineString([[start.x, start.y], [end.x, end.y]])
