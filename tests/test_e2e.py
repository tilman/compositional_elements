import os
import pickle

import cv2
from compositional_elements.generate import global_action, pose_abstraction
from compositional_elements.visualize import visualize
from compositional_elements.detect import person_detection, pose_estimation, converter

def run_before():
    script_dir = os.path.dirname(__file__)
    img_path=os.path.join(script_dir, "./test.jpg")
    img = cv2.imread(img_path)
    return script_dir, img, img_path

def test_e2e():
    script_dir, img, img_path = run_before()
    _, _, det_data = person_detection.person_detection(img_path=img_path, person_detector="Faster R-CNN")
    hrnet_output = pose_estimation.pose_estimation(detections=det_data["detections"],
                                centers=det_data["centers"],
                                scales=det_data["scales"],
                                img_path=img_path,
                                keypoint_detector="Baseline HRNet")
    poses = converter.hrnet_to_icc_poses(hrnet_output)
    global_action_lines = global_action.get_global_action_lines(poses)
    pose_lines = pose_abstraction.get_pose_lines(poses)
    img = visualize.global_action_lines(global_action_lines, img)
    img = visualize.pose_lines(pose_lines, img)
    visualize.safe(os.path.join(script_dir, "output_e2e.jpg"), img)
