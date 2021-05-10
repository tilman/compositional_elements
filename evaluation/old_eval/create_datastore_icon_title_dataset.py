# call this script with `python -m playground.create_datastore`
import os
import pickle
from pathlib import Path
import requests
import shutil

import cv2
import csv
from tqdm import tqdm
from compoelem.generate import global_action, pose_abstraction, pose_direction
from compoelem.visualize import visualize
from compoelem.detect import converter
from compoelem.detect.openpose_wrapper import get_poses
from compoelem.detect.openpose.lib.utils.common import draw_humans

# INPUT_DIRS = [
#     # "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dirk_27.11.2020_Sample_Adoration_Annunciation_Baptism/Adoration",
#     # "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dirk_27.11.2020_Sample_Adoration_Annunciation_Baptism/Annunciation",
#     # "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dirk_27.11.2020_Sample_Adoration_Annunciation_Baptism/Baptism",
#     # "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dirk_27.11.2020_Sample_Adoration_Annunciation_Baptism/Baptism",
#     # "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/old_icc/icc_images_imdahl/"
#     "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/til_selection_icon_title_dataset_2"
# ]
DATASTORE_FILE = os.path.join(os.path.dirname(__file__), "./datastore_icon_title_2.pkl")
try:
    datastore = pickle.load(open(DATASTORE_FILE, "rb"))
except FileNotFoundError as e:
    datastore = {}

def update_datastore(img_name, row, humans, poses, pose_lines, pose_directions, global_action_lines, height, width):
    datastore[img_name] = {"humans":humans, "row":row, "poses":poses, "pose_lines":pose_lines, "pose_directions":pose_directions, "height":height, "width":width, "global_action_lines":global_action_lines}
    pickle.dump(datastore, open(DATASTORE_FILE, "wb"))

def download_img(row):
    print(row)
    r = requests.get(row["url"], stream = True)
    if r.status_code == 200:
        r.raw.decode_content = True
        with open('tmp.jpg','wb') as f:
            shutil.copyfileobj(r.raw, f)
        print('Image sucessfully Downloaded:')
    else:
        print('Image Couldn\'t be retreived')
    img = cv2.imread("tmp.jpg")
    return img
def compute_non_existent_images(row):
    if(row["class"] == "still-life"):
        print("skipping still-life images")
        return
    img_name = row["image"]
    if(img_name in datastore):
        return
    img = download_img(row)
    if img is None:
        print("could not open img:", img_name)
        return
    img = converter.resize(img)
    width, height, _ = img.shape # type: ignore
    humans = get_poses(img)
    poses = converter.openpose_to_compoelem_poses(humans, *img.shape[:2]) # type: ignore
    img = draw_humans(img, humans)
    print("poses", len(poses))
    pose_directions = pose_direction.get_pose_directions(poses)
    global_action_lines = global_action.get_global_action_lines(poses)
    pose_lines = pose_abstraction.get_pose_lines(poses)
    # img = visualize.pose_directions(pose_directions, img, (255,255,100), True)
    # img = visualize.global_action_lines(global_action_lines, img)
    # img = visualize.pose_lines(pose_lines, img)    
    # visualize.safe(os.path.join(output_dir, basename), img)
    update_datastore(img_name, row, humans, poses, pose_lines, pose_directions, global_action_lines, width, height)

# images = [os.path.join(dir, file) for dir in INPUT_DIRS for file in os.listdir(dir)]
# images.sort()
# images = images[:]
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# print(len(images))
# for img_path in tqdm(images, total=len(images)):
#     compute_non_existent_images(img_path)

p = Path(__file__).with_name('data_clean.csv')
fulllist = []
with p.open('r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        fulllist.append(row)
    for row in tqdm(fulllist, total=len(fulllist)):
        compute_non_existent_images(row)
        # print(row['image'], row['url'])