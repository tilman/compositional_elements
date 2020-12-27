"""
Methods for pose estimation and keypoint detection

@author: Angel Villar-Corrales
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import DataParallel

sys.path.append("./compositional_elements/detect/lib/PoseBasedRetrievalDemo/src/API")
from compositional_elements.detect.lib.PoseBasedRetrievalDemo.src.API.lib.neural_nets.HRNet import PoseHighResolutionNet
# from compositional_elements.detect.lib.EnhancePoseEstimation.src.models.HRNet import PoseHighResolutionNet
from compositional_elements.detect.lib.PoseBasedRetrievalDemo.src.API.lib.pose_parsing import (create_pose_entries, get_final_preds_hrnet,
                              get_max_preds_hrnet)
# from compositional_elements.detect.lib.EnhancePoseEstimation.src.lib.pose_parsing import (create_pose_entries, get_final_preds_hrnet,
#                               get_max_preds_hrnet)

model = None
normalizer = None

def setup():
    global model
    if(model is None):
        print("Initializing Pose Estimation model...")
        model = PoseHighResolutionNet(is_train=False)

        print("Loading pretrained model parameters...")
        # if(estimator_name == "Baseline HRNet"):
        pretrained_path = os.path.join(os.getcwd(), "resources", "coco_hrnet_w32_256x192.pth")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = DataParallel(model)
        model = model.eval()

    # intiializing preprocessing method
    global normalizer
    if(normalizer is None):
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return model, normalizer


def get_pose_keypoints(detections, centers, scales):
    model, normalizer = setup()
    # skipping images with no person detections
    if(len(detections) == 0):
        pose_data = {
            "indep_pose_entries": np.array([]),
            "indep_all_keypoints": np.array([]),
            "pose_entries": np.array([]),
            "all_keypoints": np.array([]),
            "pose_paths": []
        }
        return pose_data

    # preprocessing the detections
    print("Preprocessing person detections...")
    norm_detections = [normalizer(torch.Tensor(det)).numpy() for det in detections]

    # forward pass through the keypoint detector model
    print("Computing forward pass through the keypoint detector model...")
    keypoint_dets = model(torch.Tensor(norm_detections).float())
    scaled_dets = F.interpolate(keypoint_dets.clone(), (256, 192),
                                mode="bilinear", align_corners=True)

    # extracting keypoint coordinates and confidence values from heatmaps
    print("Extracting keypoints from heatmaps...")
    keypoint_coords,\
        max_vals_coords = get_max_preds_hrnet(scaled_heats=scaled_dets.detach().numpy())
    keypoints, max_vals, _ = get_final_preds_hrnet(heatmaps=keypoint_dets.detach().numpy(),
                                                  center=centers, scale=scales)

    # parsing poses by combining and joining keypoits
    print("Parsing human poses...")
    indep_pose_entries, indep_all_keypoints = create_pose_entries(keypoints=keypoint_coords,
                                                                  max_vals=max_vals_coords,
                                                                  thr=0.1)
    indep_all_keypoints = [indep_all_keypoints[:, 1], indep_all_keypoints[:, 0],\
                           indep_all_keypoints[:, 2], indep_all_keypoints[:, 3]]
    indep_all_keypoints = np.array(indep_all_keypoints).T
    pose_entries, all_keypoints = create_pose_entries(keypoints=keypoints,
                                                      max_vals=max_vals,
                                                      thr=0.1)
    all_keypoints = [all_keypoints[:, 1], all_keypoints[:, 0],\
                     all_keypoints[:, 2], all_keypoints[:, 3]]
    all_keypoints = np.array(all_keypoints).T

    # creating pose visualizations and saving in the corresponding directory
    # img_name = os.path.basename(img_path)
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # create_directory(os.path.join(os.getcwd(), "data", "final_results", "pose_estimation"))
    # savepath = os.path.join(os.getcwd(), "data", "final_results", "pose_estimation", img_name)
    #draw_pose(img/255, pose_entries, all_keypoints, savefig=True, savepath=savepath)
    # pose_paths = []
    # for i, det in enumerate(detections):
    #     det_name = img_name.split(".")[0] + f"_det_{i}." + img_name.split(".")[1]
    #     savepath = os.path.join(os.getcwd(), "data", "final_results",
    #                             "pose_estimation", det_name)
    #     pose_paths.append(savepath)

    # returning pose data in correct format
    pose_data = {
        "indep_pose_entries": indep_pose_entries,
        "indep_all_keypoints": indep_all_keypoints,
        "pose_entries": pose_entries,
        "all_keypoints": all_keypoints,
        # "pose_paths": pose_paths
    }

    return pose_data