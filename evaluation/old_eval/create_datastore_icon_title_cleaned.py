#run as python module!!
import cv2
import numpy as np
import csv
from tqdm import tqdm
import os
import pickle
try:
    from compoelem.generate import global_action, pose_abstraction, pose_direction
    from compoelem.visualize import visualize
    from compoelem.detect import converter
    from compoelem.detect.openpose_wrapper import get_poses
except ModuleNotFoundError:
    print("run as python module!! -> python -m evaluation.create_datastore_icon_title_cleaned")
    exit()

# eval names:
# filterGac -> filter_pose_line_ga_result
# compare_pose_lines_2 -> cp2
# norm_by_global_action -> gacNorm
# minmax_norm_by_imgrect -> rectNorm

######################################################## config params
arch = "compoelem"
eval_arch_name = "_"+arch+"_gacNorm_cpl2_filterGac"
DATASTORE_FILE = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem/evaluation/datastore"+eval_arch_name+"_icon_title_cleaned.pkl"
DATASET_ROOT_DIR = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/evaluation/download_icon_title/dataset_cleaned/"


######################################################## model setup & download
def compute_non_existent_images(row):
    img_name = row["image"]
    if(img_name in datastore):
        return
    img = cv2.imread(DATASET_ROOT_DIR+row["class"]+"/"+row["image"])
    if img is None:
        print("could not open img:", img_name)
        return
    img = converter.resize(img)
    width, height, _ = img.shape # type: ignore
    humans = get_poses(img)
    poses = converter.openpose_to_compoelem_poses(humans, *img.shape[:2]) # type: ignore
    print("poses", len(poses))
    pose_directions = pose_direction.get_pose_directions(poses)
    global_action_lines = global_action.get_global_action_lines(poses)
    pose_lines = pose_abstraction.get_pose_lines(poses)
    update_datastore(img_name, row, humans, poses, pose_lines, pose_directions, global_action_lines, width, height)

######################################################## main loop/calculation

try:
    datastore = pickle.load(open(DATASTORE_FILE, "rb"))
except FileNotFoundError as e:
    datastore = {}

def update_datastore(img_name, row, humans, poses, pose_lines, pose_directions, global_action_lines, height, width):
    datastore[img_name] = {"humans":humans, "row":row, "poses":poses, "pose_lines":pose_lines, "pose_directions":pose_directions, "height":height, "width":width, "global_action_lines":global_action_lines}
    pickle.dump(datastore, open(DATASTORE_FILE, "wb"))

fulllist = []
with open(DATASET_ROOT_DIR+'data_clean.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = DATASET_ROOT_DIR+row["class"]+"/"+row["image"]
        if(os.path.isfile(filename)):
            fulllist.append(row)

classes = list(map(lambda x: x["class"], fulllist))
class_names, class_dist = np.unique(classes, return_counts=True)
print(class_names, class_dist)
print("\n")

for row in tqdm(fulllist, total=len(fulllist)):
    if row["image"] not in datastore:
      #feature_map = get_featuremaps(DATASET_ROOT_DIR+row["class"]+"/"+row["image"])
      #update_datastore(row, feature_map)
      compute_non_existent_images(row)


classes = list(map(lambda x: datastore[x]["row"]["class"], datastore.keys()))
class_names, class_dist = np.unique(classes, return_counts=True)
# some images can be corrupt if removed from site. This amount of pictures here are the real amount of pictures analyzed!
print(class_names, class_dist)
print("\n")