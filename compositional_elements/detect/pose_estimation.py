"""
Methods for pose estimation and keypoint detection

@author: Angel Villar-Corrales
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import DataParallel

from compositional_elements.detect.lib.neural_nets.HRNet import PoseHighResolutionNet
from compositional_elements.detect.lib.pose_parsing import (create_pose_entries, get_final_preds_hrnet,
                              get_max_preds_hrnet)

ESTIMATOR_NAME = None
KEYPOINT_ESTIMATOR = None
NORMALIZER = None

def create_directory(path):
    """
    Checking if a directory exists, and creating it if it doesnt
    """

    if(not os.path.exists(path)):
        os.makedirs(path)

    return

def setup_pose_estimator(estimator_name):
    """
    Initializing the pretrained pose estimation model
    """

    print("Initializing Pose Estimation model...")
    model = PoseHighResolutionNet(is_train=False)

    print("Loading pretrained model parameters...")
    if(estimator_name == "Baseline HRNet"):
        pretrained_path = os.path.join(os.getcwd(), "resources", "coco_hrnet_w32_256x192.pth")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = DataParallel(model)
    elif(estimator_name == "Styled HRNet"):
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "styled_hrnet_w32_256x192.pth")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(checkpoint)
    elif(estimator_name == "Tuned HRNet"):
        # TODO: replace with actual tuned HRNet model
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "styled_hrnet_w32_256x192.pth")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(checkpoint)
    model = model.eval()

    # intiializing preprocessing method
    global NORMALIZER
    NORMALIZER = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

    return model


def pose_estimation(detections, centers, scales, img_path, keypoint_detector):
    """
    Detecting the keypoints for the input images and parsing the human poses

    Args:
    -----
    detections: numpy array
        person dectections from the input images. Shape is (n_dets, 3, 256, 192)
    centers: numpy array
        center coordinates of each person detection
    scales: numpy array
        scale factor for each person detection
    img_path: string
        path to the image to extract the detections from
    keypoint_detector: string
        name of the model to use for keypoint detection ['baseline_hrnet',
        'styled_hrnet', 'tuned_hrnet']

    Returns:
    --------
    pose_data: dictionary
        dict containing the processed information about keypoints and pose objects
        for each person instance and for the full joined image
    """

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

    # initializing the model if necessary
    global KEYPOINT_ESTIMATOR
    global ESTIMATOR_NAME
    if(KEYPOINT_ESTIMATOR is None or ESTIMATOR_NAME != keypoint_detector):
        ESTIMATOR_NAME = keypoint_detector
        KEYPOINT_ESTIMATOR = setup_pose_estimator(keypoint_detector)

    # preprocessing the detections
    print("Preprocessing person detections...")
    norm_detections = [NORMALIZER(torch.Tensor(det)).numpy() for det in detections]

    # forward pass through the keypoint detector model
    print("Computing forward pass through the keypoint detector model...")
    keypoint_dets = KEYPOINT_ESTIMATOR(torch.Tensor(norm_detections).float())
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
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    create_directory(os.path.join(os.getcwd(), "data", "final_results", "pose_estimation"))
    savepath = os.path.join(os.getcwd(), "data", "final_results", "pose_estimation", img_name)
    #draw_pose(img/255, pose_entries, all_keypoints, savefig=True, savepath=savepath)
    pose_paths = []
    for i, det in enumerate(detections):
        det_name = img_name.split(".")[0] + f"_det_{i}." + img_name.split(".")[1]
        savepath = os.path.join(os.getcwd(), "data", "final_results",
                                "pose_estimation", det_name)
        pose_paths.append(savepath)

    # returning pose data in correct format
    pose_data = {
        "indep_pose_entries": indep_pose_entries,
        "indep_all_keypoints": indep_all_keypoints,
        "pose_entries": pose_entries,
        "all_keypoints": all_keypoints,
        "pose_paths": pose_paths
    }

    return pose_data
