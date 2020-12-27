# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import os
import sys
# sys.path.append("..")

import cv2
from tqdm import tqdm
from compositional_elements.generate import global_action, pose_abstraction, pose_direction
from compositional_elements.visualize import visualize
from compositional_elements.detect import converter
from compositional_elements.detect.faster_rcnn import get_person_boundingboxes
from compositional_elements.detect.hrnet import get_pose_keypoints
# from compositional_elements.detect.lib.EnhancePoseEstimation.src.lib import person_detection, pose_estimation
# sys.path.append("./compositional_elements/detect/lib/PoseBasedRetrievalDemo/src/API")
# from compositional_elements.detect.lib.PoseBasedRetrievalDemo.src.API.lib import person_detection, pose_estimation
INPUT_DIR = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/old_icc/icc_images_imdahl/"
OUTPUT_DIR = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/evaluation/compositional_elements/v0.0.2_27.12.20/icc_images_imdahl"

def get_icc(output_dir, img_path):
    basename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    # _, _, det_data = person_detection.person_detection(img_path=img_path, person_detector="Faster R-CNN")
    # hrnet_output = pose_estimation.pose_estimation(detections=det_data["detections"],
    #                             centers=det_data["centers"],
    #                             scales=det_data["scales"],
    #                             img_path=img_path,
    #                             keypoint_detector="Baseline HRNet")
    person_boundingboxes = get_person_boundingboxes(img_path)
    # debug test: is person_boundingboxes["detections"] already an image

    # _, _, det_data = person_detection.person_detection(img_path=img_path, person_detector="Faster R-CNN")
    hrnet_output = get_pose_keypoints(detections=person_boundingboxes["detections"],
                                centers=person_boundingboxes["centers"],
                                scales=person_boundingboxes["scales"])
    poses = converter.hrnet_to_icc_poses(hrnet_output)
    global_action_lines = global_action.get_global_action_lines(poses)
    pose_lines = pose_abstraction.get_pose_lines(poses)
    pose_directions = pose_direction.get_pose_directions(poses)
    img = visualize.global_action_lines(global_action_lines, img)
    img = visualize.pose_lines(pose_lines, img)
    img = visualize.poses(poses, img)
    img = visualize.pose_directions(pose_directions, img, (0, 200, 255))
    visualize.safe(os.path.join(output_dir, basename), img)

images = [os.path.join(os.getcwd(), INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f[0] != '.'][:]
images.sort()
images = images[:]
os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.chdir(OUTPUT_DIR)
print(len(images))
for img_path in tqdm(images, total=len(images)):
    get_icc(OUTPUT_DIR, img_path)