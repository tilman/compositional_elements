import os
import pickle

import cv2
from compoelem.detect import converter, openpose_wrapper
from compoelem.generate import pose_abstraction
from compoelem.visualize import visualize
from compoelem.compare.pose_line import compare_pose_lines


def prepare(img_path):
    img = cv2.imread(img_path)
    img = converter.resize(img)
    humans = openpose_wrapper.get_poses(img)
    poses = converter.openpose_to_compoelem_poses(humans, *img.shape[:2])
    pose_lines = pose_abstraction.get_pose_lines(poses)
    img = visualize.pose_lines(pose_lines, img)
    return img, poses, pose_lines

def test_compare_two_similar_images():
    script_dir = os.path.dirname(__file__)
    img_a, poses_a, pose_lines_a = prepare(os.path.join(script_dir, "./test_query.jpg"))
    img_b, poses_b, pose_lines_b = prepare(os.path.join(script_dir, "./test_similar.jpg"))
    similarity = compare_pose_lines(pose_lines_a, pose_lines_b)
    print("similarity", similarity) # 1261.975393830468
    visualize.safe(os.path.join(script_dir, "output_test_query.jpg"), img_a)
    visualize.safe(os.path.join(script_dir, "output_test_similar.jpg"), img_b)

def test_compare_two_unsimilar_images():
    script_dir = os.path.dirname(__file__)
    img_a, poses_a, pose_lines_a = prepare(os.path.join(script_dir, "./test_query.jpg"))
    img_b, poses_b, pose_lines_b = prepare(os.path.join(script_dir, "./test_unsimilar.jpg"))
    similarity = compare_pose_lines(pose_lines_a, pose_lines_b) # 1491.137705564878
    print("similarity", similarity)
    visualize.safe(os.path.join(script_dir, "output_test_query.jpg"), img_a)
    visualize.safe(os.path.join(script_dir, "output_test_unsimilar.jpg"), img_b)