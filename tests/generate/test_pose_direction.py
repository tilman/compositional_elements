import pickle
import cv2
import os
from compositional_elements.generate import pose_direction
from compositional_elements.detect import converter
from compositional_elements.visualize import visualize


def test_get_pose_directions():
    print("execute test_get_pose_directions")
    script_dir = os.path.dirname(__file__)
    file_path=os.path.join(script_dir, "../test.jpg")
    pose_data = pickle.load(open(os.path.join(script_dir, "../test_pose_data.p"), "rb"))
    pose_direction.get_pose_directions()
    img = cv2.imread(file_path)
    poses = converter.hrnet_to_icc_poses(pose_data)
    pose_directions = pose_direction.get_pose_directions(poses)
    img = visualize.pose_directions(pose_directions, img)
    print("visualize test_get_pose_directions")
    visualize.draw_window('pose', img)