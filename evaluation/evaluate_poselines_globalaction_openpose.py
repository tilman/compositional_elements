# call this script with `python -m evaluation.evaluate_poselines_globalaction`
import os

import cv2
from tqdm import tqdm
from compoelem.generate import global_action, pose_abstraction, pose_direction
from compoelem.visualize import visualize
from compoelem.detect import converter
from compoelem.detect.openpose_wrapper import get_poses
from compoelem.detect.openpose.lib.utils.common import draw_humans

INPUT_DIR = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/old_icc/icc_images_imdahl/"
OUTPUT_DIR = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/evaluation/compoelem/v0.0.6_v0.0.6_conesc30_coneop80_negGlobalAngle_12.01.21/icc_images_imdahl"

def get_icc(output_dir, img_path):
    basename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    img = converter.resize(img)
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
    visualize.safe(os.path.join(output_dir, basename), img)

images = [os.path.join(os.getcwd(), INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f[0] != '.'][:]
images.sort()
images = images[:]
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(len(images))
for img_path in tqdm(images, total=len(images)):
    get_icc(OUTPUT_DIR, img_path)