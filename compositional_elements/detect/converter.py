from typing import Tuple

from .types import *


def hrnet_to_icc_poses(pose_data) -> Poses:
    poses: Poses = []
    for entry in pose_data["pose_entries"]:
        keypoints: Sequence[Keypoint] = []
        for keypoint in entry:
            if keypoint != -1:
                try:
                    y, x = list(pose_data["all_keypoints"][int(keypoint)][0:2])
                    keypoint = Keypoint(int(x), int(y))
                    keypoints.append(keypoint)
                except IndexError:
                    print('could not add keypoint')
            else:
                keypoints.append(NoneKeypoint())
        pose = Pose(keypoints)
        poses.append(pose)
    return poses

def p(point: Point) -> Tuple[int, int]:
    return (int(point.x), int(point.y))

def k(keypoint: Keypoint) -> Tuple[int, int]:
    return (int(keypoint.x), int(keypoint.y))
