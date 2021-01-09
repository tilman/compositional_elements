from compoelem.detect.openpose.lib.utils.common import BodyPart, Human
from typing import Tuple

from compoelem.types import *


def hrnet_to_compoelem_poses(pose_data) -> Poses:
    poses: Poses = []
    for entry in pose_data["pose_entries"]:
        keypoints: Sequence[Keypoint] = []
        for ik, keypoint in enumerate(entry):
            if keypoint != -1 and ik != 18 and ik != 17:
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

def body_part_to_keypoint(body_part: BodyPart, img_height: int, img_width: int) -> Keypoint:
    x = int(body_part.x * img_width + 0.5)
    y = int(body_part.y * img_width + 0.5)
    return Keypoint(x, y, body_part.score)

def openpose_to_compoelem_poses(humans: Sequence[Human], img_height: int, img_width: int) -> Poses:
    poses: Poses = []
    for human in humans:
        keypoints = [body_part_to_keypoint(human.body_parts[i], img_height, img_width) if i in human.body_parts else NoneKeypoint() for i in range(0, 18)]
        pose = Pose(keypoints)
        poses.append(pose)
    return poses

def p(point: Point) -> Tuple[int, int]:
    return (int(point.x), int(point.y))

def k(keypoint: Keypoint) -> Tuple[int, int]:
    return (int(keypoint.x), int(keypoint.y))
