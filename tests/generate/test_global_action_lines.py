import os
import pickle

import cv2
from compoelem.detect import converter
from compoelem.generate import global_action, pose_direction
from compoelem.visualize import visualize

def run_before():
    script_dir = os.path.dirname(__file__)
    img_path=os.path.join(script_dir, "../test.jpg")
    img = cv2.imread(img_path)
    pose_data = pickle.load(open(os.path.join(script_dir, "./test_humans_data.p"), "rb"))
    poses = converter.openpose_to_compoelem_poses(pose_data, *img.shape[:2])
    return script_dir, img, poses

def test_get_global_action_lines():
    script_dir, img, poses = run_before()
    global_action_lines = global_action.get_global_action_lines(poses)
    #print(global_action_lines)
    #img = visualize.poses(poses, img)
    #pose_directions = pose_direction.get_pose_directions(poses)
    #img = visualize.pose_directions(pose_directions, img, (0,255,255), True)
    img = visualize.global_action_lines(global_action_lines, img)
    visualize.safe(os.path.join(script_dir, "output_test_get_global_action_lines.jpg"), img)
