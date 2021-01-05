import os
import pickle

from torchopenpose import util
import cv2

from compoelem.generate import global_action, pose_abstraction
from compoelem.visualize import visualize
from compoelem.detect import converter
# from compoelem.detect.faster_rcnn import get_person_boundingboxes
# from compoelem.detect.hrnet import get_pose_keypoints
from compoelem.detect.openpose import get_poses

def run_before():
    script_dir = os.path.dirname(__file__)
    img_path=os.path.join(script_dir, "./test.jpg")
    img = cv2.imread(img_path)
    return script_dir, img, img_path

def test_e2e():
    script_dir, img, img_path = run_before()
    candidate, subset = get_poses(img)
    img = util.draw_bodypose(img, candidate, subset)
    visualize.safe(os.path.join(script_dir, "output_e2e.jpg"), img)
    print(candidate, subset)
    # person_boundingboxes = get_person_boundingboxes(img_path)
    # # _, _, det_data = person_detection.person_detection(img_path=img_path, person_detector="Faster R-CNN")
    # hrnet_output = get_pose_keypoints(detections=person_boundingboxes["detections"],
    #                             centers=person_boundingboxes["centers"],
    #                             scales=person_boundingboxes["scales"])
    #                             # img_path=img_path,
    #                             # keypoint_detector="Baseline HRNet")
    # poses = converter.hrnet_to_icc_poses(hrnet_output)

    # global_action_lines = global_action.get_global_action_lines(poses)
    # pose_lines = pose_abstraction.get_pose_lines(poses)
    # img = visualize.global_action_lines(global_action_lines, img)
    # img = visualize.boundingboxes(person_boundingboxes["boxes"][0], person_boundingboxes["scores"][0], img)
    # img = visualize.pose_lines(pose_lines, img)
