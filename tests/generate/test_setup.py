import cv2
import os
import pickle
from compoelem.detect import converter

def run_before():
    script_dir = os.path.dirname(__file__)
    img_path=os.path.join(script_dir, "../test.jpg")
    img = cv2.imread(img_path)
    pose_data = pickle.load(open(os.path.join(script_dir, "../test_pose_data.p"), "rb"))
    poses = converter.hrnet_to_icc_poses(pose_data)
    return script_dir, img, poses