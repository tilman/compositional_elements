import os
import cv2

from compoelem.generate import global_action, pose_abstraction, pose_direction
from compoelem.visualize import visualize
from compoelem.detect import converter
from compoelem.detect.openpose_wrapper import get_poses
from compoelem.detect.openpose.lib.utils.common import draw_humans

def run_before():
    script_dir = os.path.dirname(__file__)
    img_path=os.path.join(script_dir, "./test.jpg")
    img = cv2.imread(img_path)
    return script_dir, img, img_path

def test_e2e():
    script_dir, img, img_path = run_before()
    humans = get_poses(img)
    poses = converter.openpose_to_compoelem_poses(humans, *img.shape[:2])
    img = draw_humans(img, humans)
    print(poses)
    pose_directions = pose_direction.get_pose_directions(poses)
    global_action_lines = global_action.get_global_action_lines(poses)
    pose_lines = pose_abstraction.get_pose_lines(poses)
    img = visualize.pose_directions(pose_directions, img, (255,255,100), True)
    img = visualize.global_action_lines(global_action_lines, img)
    img = visualize.pose_lines(pose_lines, img)
    visualize.safe(os.path.join(script_dir, "output_e2e.jpg"), img)
