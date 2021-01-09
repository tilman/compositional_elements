import os
import pickle

import cv2
from compoelem.detect import converter
from compoelem.generate import pose_abstraction
from compoelem.visualize import visualize


def run_before():
    script_dir = os.path.dirname(__file__)
    img_path=os.path.join(script_dir, "../test.jpg")
    img = cv2.imread(img_path)
    pose_data = pickle.load(open(os.path.join(script_dir, "./test_humans_data.p"), "rb"))
    poses = converter.openpose_to_compoelem_poses(pose_data, *img.shape[:2])
    return script_dir, img, poses

def test_get_pose_lines():
    script_dir, img, poses = run_before()
    pose_lines = pose_abstraction.get_pose_lines(poses)
    img = visualize.pose_lines(pose_lines, img)
    visualize.safe(os.path.join(script_dir, "output_test_get_pose_lines.jpg"), img)

def test_get_pose_triangles():
    script_dir, img, poses = run_before()
    pose_triangles = pose_abstraction.get_pose_triangles(poses)
    img = visualize.pose_triangles(pose_triangles, img)
    visualize.safe(os.path.join(script_dir, "output_test_get_pose_triangles.jpg"), img)
