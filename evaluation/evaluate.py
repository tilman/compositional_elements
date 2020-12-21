import os
import pickle
import tqdm

import cv2
from compositional_elements.generate import global_action, pose_abstraction
from compositional_elements.visualize import visualize
from compositional_elements.detect import person_detection, pose_estimation, converter

INPUT_DIR = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/old_icc/icc_images_imdahl/"
OUTPUT_DIR = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/evaluation/compositional_elements/v0.0.1_21.12.20/icc_images_imdahl"

def get_icc(output_dir, img_path):
    img = cv2.imread(img_path)
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
    visualize.safe(os.path.join(output_dir, "output_e2e.jpg"), img)

images = [os.path.join(os.getcwd(), INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f[0] != '.'][START_INDEX:]
images.sort()
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)
print(len(images))
for img_name in tqdm(images, total=len(images)):
    basename = os.path.basename(img_name)