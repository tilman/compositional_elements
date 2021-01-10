import itertools
from typing import Tuple, cast

import numpy as np
import numpy.linalg as la
from shapely.geometry import Polygon

from compoelem.config import config
from compoelem.detect.converter import k, p
from compoelem.types import *

CORRECTION_ANGLE=config["bisection"]["correction_angle"]
CONE_OPENING_ANGLE=config["bisection"]["cone_opening_angle"]
CONE_SHAPE_LENGTH=int(config["bisection"]["cone_shape_length"])

def get_centroids_for_bisection(keypoints: Sequence[Keypoint]) -> Tuple[Keypoint, Keypoint, Keypoint]:
    """Helper method for transforming HRNet input in a way that we can calculate the bisection vector of upper,
    middle and lower keypoints. Therefore we calculate the centroid of theses pairs: (4,3) (6,10) (12,11)

    Args:
        keypoints (Pose.keypoints): transformed HRNet output

    Raises:
        ValueError: is raised if one of the above keypoints is missing

    Returns:
        Tuple[Keypoint, Keypoint, Keypoint]: top_kp, middle_kp, bottom_kp
    """
    bisection_keypoint_pairs: Sequence[Tuple[Keypoint,Keypoint]] = list(
        filter(lambda x: not x[0].isNone or not x[1].isNone,
            zip(
                np.array(keypoints)[config["bisection"]["left_pose_points"]], 
                np.array(keypoints)[config["bisection"]["right_pose_points"]]
            )
        )
    )
    if len(bisection_keypoint_pairs) != 3:
        raise ValueError('some keypoints for bisection calculation are missing!')
    keypoint_pairs = [Keypoint(*p(cast(Point, LineString([k(a),k(b)]).centroid)), np.mean([a.score, b.score])) for a,b in bisection_keypoint_pairs]
    top_kp, middle_kp, bottom_kp = keypoint_pairs
    return top_kp, middle_kp, bottom_kp

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


# previously poseToBisectCone
def get_bisect_cone(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> Polygon:
    a = np.array([top_kp.x, top_kp.y])
    b = np.array([middle_kp.x, middle_kp.y])
    c = np.array([bottom_kp.x, bottom_kp.y])
    length: int = CONE_SHAPE_LENGTH
    width = np.deg2rad(CONE_OPENING_ANGLE)
    angle = getAngleGroundNormed(a,b,c)
    conePoint1 = (int(length * np.cos(angle - (width/2))), int(length * np.sin(angle - (width/2)))) #with origin zero
    conePoint1 = (b[0] + conePoint1[0], b[1] - conePoint1[1])
    conePoint2 = (int(length * np.cos(angle+(width/2))), int(length * np.sin(angle + (width/2)))) #with origin zero
    conePoint2 = (b[0] + conePoint2[0], b[1] - conePoint2[1])
    cone = Polygon([conePoint1, conePoint2, b])
    return cone

def getAngle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

# make it more simple by calculating getAngle for both directions (a,b,c and c,b,a and take the smaller result)
def getAngleGroundNormed(a,b,c):
    # check if the angle is to the left or the right side
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


#previuosly angleMapper
def get_mapped_angle(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> float:
    a = np.array([top_kp.x, top_kp.y])
    b = np.array([middle_kp.x, middle_kp.y])
    c = np.array([bottom_kp.x, bottom_kp.y])
    angle = getAngleGroundNormed(a,b,c)
     #map all angles to one direction, so the mean does not get influences by left/right direction
    if(angle > np.deg2rad(180)):
        mapped = angle - np.deg2rad(180)
        return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped
    else:
        mapped = angle
        return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped

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

def poseToBisectVector(pose):
    points = pose[[0,1,8]]
    if(0.0 in points[:,2:]): #if one point has confidence zero, we can not generate the vector
        return None
    a,b,c = points[:,:2] # cut of confidence score so we have normal coordinate points
    bisecPoint = getBisecPoint(a,b,c)
    return np.array([bisecPoint,b])