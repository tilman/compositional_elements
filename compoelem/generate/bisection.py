import itertools
from typing import Tuple, cast

import numpy as np
import numpy.linalg as la
from shapely.geometry import Polygon

from compoelem.config import config
from compoelem.detect.converter import k, p
from compoelem.types import *

body_direction_counter = 0
normal_body_direction_counter = 0
fallback_body_direction_counter = 0

def get_centroids_for_bisection(keypoints: Sequence[Keypoint], fallback: bool) -> Tuple[Keypoint, Keypoint, Keypoint]:
    """Helper method for transforming COCO input in a way that we can calculate the bisection vector of upper,
    middle and lower keypoints. Therefore we calculate the centroid of the keypoint pairs specified in 
    the config bisection.left_pose_points and bisection.right_pose_points

    Args:
        keypoints (Pose.keypoints): transformed COCO output

    Raises:
        ValueError: is raised if one of the above keypoints is missing

    Returns:
        Tuple[Keypoint, Keypoint, Keypoint]: top_kp, middle_kp, bottom_kp
    """
    global body_direction_counter
    global normal_body_direction_counter
    global fallback_body_direction_counter
    body_direction_counter = body_direction_counter + 1

    keypoints_np = np.array(keypoints)
    left_kp = keypoints_np[config["bisection"]["left_pose_points"]]
    right_kp = keypoints_np[config["bisection"]["right_pose_points"]]

    left_shoulder_kp = keypoints_np[config["bisection"]["fallback"]["left_shoulder_kp"]]
    right_shoulder_kp = keypoints_np[config["bisection"]["fallback"]["right_shoulder_kp"]]
    left_eye_kp = keypoints_np[config["bisection"]["fallback"]["left_eye_kp"]]
    right_eye_kp = keypoints_np[config["bisection"]["fallback"]["right_eye_kp"]]
    with_fallback_kps: Sequence[Tuple[Keypoint,Keypoint]] = []
    if fallback:
        fb_used = False
        for idx, t in enumerate(zip(left_kp, right_kp)): # 0,1,8 & 11
            if t[0].isNone or t[1].isNone:
                fb_used = True
                if idx == 0: # top point fallback
                    with_fallback_kps.append((left_eye_kp, right_eye_kp))
                if idx == 1: # middle point fallback
                    with_fallback_kps.append((left_shoulder_kp, right_shoulder_kp))
                if idx == 2: # bottom point fallback
                    l_middle_point_pair, r_middle_point_pair = with_fallback_kps[-1]
                    left_bottom_fallback = Keypoint(l_middle_point_pair.x, l_middle_point_pair.y + 40, 1)
                    right_bottom_fallback = Keypoint(r_middle_point_pair.x, r_middle_point_pair.y + 40, 1)
                    with_fallback_kps.append((left_bottom_fallback, right_bottom_fallback))
            else:
                with_fallback_kps.append(t)
        if fb_used:
            fallback_body_direction_counter = fallback_body_direction_counter + 1
        else:
            normal_body_direction_counter = normal_body_direction_counter + 1

    bisection_keypoint_pairs: Sequence[Tuple[Keypoint,Keypoint]] = list(
        filter(lambda x: not (x[0].isNone or x[1].isNone), with_fallback_kps if fallback else zip(left_kp, right_kp,))
    )
    if len(bisection_keypoint_pairs) != 3:
        raise ValueError('some keypoints for bisection calculation are missing!')
    keypoint_pairs = [
        Keypoint(*p(cast(Point, LineString([k(a),k(b)]).centroid)), np.mean([a.score, b.score]))
        for a,b in bisection_keypoint_pairs
    ]
    top_kp, middle_kp, bottom_kp = keypoint_pairs
    # print("body_direction_counter, normal_body_direction_counter, fallback_body_direction_counter", body_direction_counter, normal_body_direction_counter, fallback_body_direction_counter)
    return top_kp, middle_kp, bottom_kp

def keypoint_to_point(k: Keypoint) -> Point:
    return Point(k.x, k.y)

def keypoint_to_vector(a: Keypoint, b: Keypoint) -> np.ndarray:
    return keypoint_to_np(a) - keypoint_to_np(b)

def keypoint_to_np(keypoint: Keypoint) -> np.ndarray:
    return np.array([keypoint.x, keypoint.y])

def get_bisection_point(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> Keypoint:
    """Returns the end point of the bisection vector of three input points.
    The length of the vector is the double of the length from top_kp to middle_kp.

    Args:
        top_kp (Keypoint): Some keypoint from head region
        middle_kp (Keypoint): Some keypoint from upper body region
        bottom_kp (Keypoint): Some keypoint from lower body region

    Returns:
        Keypoint: Endpoint of bisection vector
    """
    phi = get_angle(top_kp, middle_kp, bottom_kp) - np.deg2rad(int(config["bisection"]["correction_angle"]))
    return get_bisection_point_from_angle(top_kp, middle_kp, phi)

def get_bisection_point_from_angle(top_kp: Keypoint, middle_kp: Keypoint, phi: float, scale: float = 1) -> Keypoint:
    theta = phi / 2 * -1 
    x, y = keypoint_to_vector(top_kp, middle_kp)
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return Keypoint(x_new * scale + middle_kp.x, y_new * scale + middle_kp.y)

def get_angle(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """Get angle between vector b->a and b->c by calculating the scalarproduct.
    Because the scalarproduct looses track of the orientation of the angle we further
    calculate the orientation with the help of the crossproduct and return it with the sign of the returned angle

    Args:
        a (Keypoint): first keypoint
        b (Keypoint): middle keypoint
        c (Keypoint): last keypoint

    Returns:
        float: angle in radians. Positive if angle is left rotating and positiv if right rotating
    """
    # get a vector with origin in (0,0) from points a and b by substracting Point a from Point b
    vector_a = keypoint_to_vector(a, b)
    vector_c = keypoint_to_vector(c, b)
    # https://de.wikipedia.org/wiki/Skalarprodukt => winkel phi = arccos(...)
    phi = np.arccos(np.dot(vector_a, vector_c) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_c)))
    angle_left_opening = np.cross(vector_a, vector_c) < 0
    return phi if angle_left_opening else -phi

def get_horizantal_b_reference(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> Keypoint:
    """Returns an reference Keypoint by adding or substracting 100px from middle_kp x-coordinate.
    If the smaller angle between the three keypoints is on the left side we substract 100px else we add 100px.
    Using this keypoint together with the middle_kp to create a parallel line to the x axis

    Args:
        top_kp (Keypoint): Some keypoint from head region
        middle_kp (Keypoint): Some keypoint from upper body region
        bottom_kp (Keypoint): Some keypoint from lower body region

    Returns:
        Keypoint: returns an reference Keypoint.
    """
    vector_a = keypoint_to_vector(top_kp, middle_kp)
    vector_c = keypoint_to_vector(bottom_kp, middle_kp)
    angle_left_opening = np.cross(vector_a, vector_c) < 0
    return Keypoint(middle_kp.x - 100, middle_kp.y) if angle_left_opening else Keypoint(middle_kp.x + 100, middle_kp.y)

# TODO: rewrite
# # previously poseToBisectCone
def get_bisection_cone(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> Polygon:
    cone_offset_angle = np.deg2rad(int(config["bisection"]["cone_opening_angle"])/2)
    cone_scale_factor = float(config["bisection"]["cone_scale_factor"])
    cone_base_scale_factor = float(config["bisection"]["cone_base_scale_factor"])
    # print(cone_offset_angle, cone_scale_factor)
    phi = get_angle(top_kp, middle_kp, bottom_kp) - np.deg2rad(int(config["bisection"]["correction_angle"]))
    # cone points
    cone_endpoint1 = get_bisection_point_from_angle(top_kp, middle_kp, phi + cone_offset_angle, cone_scale_factor)
    cone_endpoint2 = get_bisection_point_from_angle(top_kp, middle_kp, phi - cone_offset_angle, cone_scale_factor)
    cone_startpoint1 = get_bisection_point_from_angle(top_kp, middle_kp, phi + np.deg2rad(225), cone_base_scale_factor)
    cone_startpoint2 = get_bisection_point_from_angle(top_kp, middle_kp, phi - np.deg2rad(225), cone_base_scale_factor)
    cone = Polygon([keypoint_to_np(cone_endpoint1), keypoint_to_np(cone_endpoint2), keypoint_to_np(cone_startpoint2), keypoint_to_np(cone_startpoint1)])
    # if phi > 0:
    #     cone = Polygon([keypoint_to_np(cone1), keypoint_to_np(cone2), [middle_kp.x, middle_kp.y - 20], [middle_kp.x, middle_kp.y + 20]]) # new
    # else:
    #     cone = Polygon([keypoint_to_np(cone1), keypoint_to_np(cone2), [middle_kp.x, middle_kp.y + 20], [middle_kp.x, middle_kp.y - 20]]) # new
    return cone

def get_angle_in_respect_to_x(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> float:
    """Calculates the angle from the three input keypoints in respect to the x-axis.
    A positive value indicates a increasing slope and a negative a decreasing slope.

    Args:
        top_kp (Keypoint): Some keypoint from head region
        middle_kp (Keypoint): Some keypoint from upper body region
        bottom_kp (Keypoint): Some keypoint from lower body region

    Returns:
        float: angle in radians in respect to x-axis
    """
    bisect_point = get_bisection_point(top_kp, middle_kp, bottom_kp)
    horizontal_middle_kp_reference = get_horizantal_b_reference(top_kp, middle_kp, bottom_kp)
    gamma = get_angle(horizontal_middle_kp_reference, middle_kp, bisect_point)
    return gamma

def keypoint_to_np(keypoint: Keypoint) -> np.ndarray:
    return np.array([keypoint.x, keypoint.y])

# #previuosly angleMapper
# def get_mapped_angle(top_kp: Keypoint, middle_kp: Keypoint, bottom_kp: Keypoint) -> float:
#     a = np.array([top_kp.x, top_kp.y])
#     b = np.array([middle_kp.x, middle_kp.y])
#     c = np.array([bottom_kp.x, bottom_kp.y])
#     angle = getAngleGroundNormed(a,b,c)
#      #map all angles to one direction, so the mean does not get influences by left/right direction
#     if(angle > np.deg2rad(180)):
#         mapped = angle - np.deg2rad(180)
#         return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped
#     else:
#         mapped = angle
#         return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped

# def angleMapper(pose):
#     angle = getAngleGroundNormed(*pose[[0,1,8]][:,:2])
#      #map all angles to one direction, so the mean does not get influences by left/right direction
#     if(angle > np.deg2rad(180)):
#         mapped = angle - np.deg2rad(180)
#         return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped
#     else:
#         mapped = angle
#         return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped

# def getGlobalLineAngle(poses):
#     return np.mean([angleMapper(pose) for pose in poses if not 0.0 in pose[[0,1,8]][:,2:]])

# def getBisecPoint(a,b,c) -> Tuple[int, int]:
#     angle = getAngleGroundNormed(a,b,c)
#     dist = la.norm(a-b)*2
#     d = (int(dist * np.cos(angle)), int(dist * np.sin(angle))) #with origin zero
#     out = (b[0]+d[0],b[1]-d[1])
#     return out #with origin b

# def poseToBisectVector(pose):
#     points = pose[[0,1,8]]
#     if(0.0 in points[:,2:]): #if one point has confidence zero, we can not generate the vector
#         return None
#     a,b,c = points[:,:2] # cut of confidence score so we have normal coordinate points
#     bisecPoint = getBisecPoint(a,b,c)
#     return np.array([bisecPoint,b])