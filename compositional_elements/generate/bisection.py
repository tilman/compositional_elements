import itertools
from typing import Tuple

import numpy as np
import numpy.linalg as la
from shapely.geometry import Polygon

from compositional_elements.config import config
from compositional_elements.types import *

CORRECTION_ANGLE=config["bisection"]["correction_angle"]
CONE_OPENING_ANGLE=config["bisection"]["cone_opening_angle"]

def get_bisection_point(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> Point:
    """Returns the end point of the bisection vector of three input points. The length of the vector is the double of the length from top_kp to middle_kp.

    Args:
        top_kp (Keypoint): Some keypoint from head region
        middle_kp (Keypoint): Some keypoint from upper body region
        bottom_kp (Keypoint): Some keypoint from lower body region

    Returns:
        Point: Endpoint of bisection vector
    """
    a = np.array([top_kp.x, top_kp.y])
    b = np.array([middle_kp.x, middle_kp.y])
    c = np.array([bottom_kp.x, bottom_kp.y])
    r = getBisecPoint(a,b,c)
    return Point(r[0], r[1])

def getAngle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

# make it more simple by calculating getAngle for both directions (a,b,c and c,b,a and take the smaller result)
def getAngleGroundNormed(a,b,c):
    if(a[0]-b[0]<0):
        b_plane_point = np.array([b[0]-50,b[1]])
    else:
        b_plane_point = np.array([b[0]+50,b[1]])
    vector_angle = getAngle(a,b,c)
    ground_angle = getAngle(a,b,b_plane_point)
    normed_angle = vector_angle/2 - ground_angle
    if(a[0]-b[0]<0):
        return (normed_angle+np.deg2rad(180-CORRECTION_ANGLE))
    else:
        return (np.deg2rad(360+CORRECTION_ANGLE)-normed_angle)

def angleMapper(pose):
    angle = getAngleGroundNormed(*pose[[0,1,8]][:,:2])
     #map all angles to one direction, so the mean does not get influences by left/right direction
    if(angle > np.deg2rad(180)):
        mapped = angle - np.deg2rad(180)
        return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped
    else:
        mapped = angle
        return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped

def getGlobalLineAngle(poses):
    return np.mean([angleMapper(pose) for pose in poses if not 0.0 in pose[[0,1,8]][:,2:]])

def getBisecPoint(a,b,c) -> Tuple[int, int]:
    angle = getAngleGroundNormed(a,b,c)
    dist = la.norm(a-b)*2
    d = (int(dist * np.cos(angle)), int(dist * np.sin(angle))) #with origin zero
    out = (b[0]+d[0],b[1]-d[1])
    return out #with origin b

def getBisecCone(a,b,c, length):
    width = np.deg2rad(CONE_OPENING_ANGLE)
    angle = getAngleGroundNormed(a,b,c)
    conePoint1 = (int(length * np.cos(angle-(width/2))), int(length * np.sin(angle-(width/2)))) #with origin zero
    conePoint2 = (int(length * np.cos(angle+(width/2))), int(length * np.sin(angle+(width/2)))) #with origin zero
    return ((b[0]+conePoint1[0],b[1]-conePoint1[1]),(b[0]+conePoint2[0],b[1]-conePoint2[1]))

def poseToBisectVector(pose):
    points = pose[[0,1,8]]
    if(0.0 in points[:,2:]): #if one point has confidence zero, we can not generate the vector
        return None
    a,b,c = points[:,:2] # cut of confidence score so we have normal coordinate points
    bisecPoint = getBisecPoint(a,b,c)
    return np.array([bisecPoint,b])

def poseToBisectCone(pose, length):
    points = pose[[0,1,8]]
    if(0.0 in points[:,2:]): #if one point has confidence zero, we can not generate the vector
        return None
    a,b,c = points[:,:2] # cut of confidence score so we have normal coordinate points
    conePoint1, conePoint2 = getBisecCone(a,b,c, length)
    return Polygon([conePoint1, conePoint2, b])

def coneIntersections(bisecCones):
    out = {}
    for r in range(1,len(bisecCones)+1):
        pc = list(itertools.combinations(range(0,len(bisecCones)),r))
        for combi in pc:
            intersect = bisecCones[combi[0]]
            for i in combi[1:]:
                intersect = intersect.intersection(bisecCones[i])
            if not intersect.is_empty:
                out[combi] = intersect
    return out
