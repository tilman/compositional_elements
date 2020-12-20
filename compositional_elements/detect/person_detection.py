"""
Initialization of the person detector model and methods for detection

@author: Angel Villar-Corrales
"""

import os

import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from lib.logger import log_function, print_
from lib.neural_nets.EfficientDet import EfficientDetBackbone as EfficientDet
from lib.neural_nets.HRNet import PoseHighResolutionNet
from lib.transforms import TransformDetection
from lib.utils import create_directory
from lib.visualizations import visualize_bbox, visualize_img

DETECTOR_NAME = None
DETECTOR = None
DETS_EXTRACTOR = None

@log_function
def setup_detector(detector_name):
    """
    Initializing person detector and loading pretrained model paramters
    """

    # intializing model skeleton for the given model type
    print_(f"Initializing Person Detector: {detector_name}")
    if(detector_name in ["Faster R-CNN", "Tuned R-CNN"]):
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    elif(detector_name in ["EfficientDet"]):
        compound_coef = 0
        anchors_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        anchors_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        model = EfficientDet(compound_coef=compound_coef, num_classes=1,
                             ratios=anchors_ratios, scales=anchors_scales,
                             threshold=0.5, iou_threshold=0.5)
    model.eval()

    # loading pretrained weights
    print_("Loading detector pretrained parameters...")
    if(detector_name == "Faster R-CNN"):
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "coco_faster_rcnn.pth")
        print_(f"    Loading: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
    elif(detector_name == "Tuned R-CNN"):
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "arch_faster_rcnn.pth")
        print_(f"    Loading: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
    elif(detector_name == "EfficientDet"):
        pretrained_path = os.path.join(os.getcwd(), "resources", "coco_efficientdet_d0.pth")
        print_(f"    Loading: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # model = DataParallel(model)

    model.load_state_dict(checkpoint)
    model = model.eval()

    # intiializing object for extracting detections
    global DETS_EXTRACTOR
    DETS_EXTRACTOR = TransformDetection(det_width=192, det_height=256)

    return model


@log_function
def person_detection(img_path, person_detector):
    """
    Computing a forward pass throught the model to detect the persons in the image

    Args:
    -----
    img_path: string
        path to the image to extract the detections from
    person_detector: string
        name of the person detector model to use ['faster_rcnn', 'efficientdet',
        'tuned-rcnn']

    Returns:
    --------
    savepath: string
        path where the image with the detected instances and bounding boxes is stored
    det_paths: list
        list with the path where the person detection images are stored
    data: dictionary
        dict containing the results from the detector, including the croped detections
        and the global position of the detection in a (center, scale) format.
        Remember detections have fixed size of (256, 192)
    """

    global DETECTOR
    global DETECTOR_NAME
    if(DETECTOR is None or DETECTOR_NAME != person_detector):
        DETECTOR_NAME = person_detector
        DETECTOR = setup_detector(detector_name=person_detector)

    # loading image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = torch.Tensor(img.transpose(2,0,1)[np.newaxis,:])

    # forward pass through person detector
    print_("Computing forward pass through person detector...")
    outputs = DETECTOR(img / 255)
    boxes, labels, scores = bbox_filtering(outputs, filter_=1, thr=0.4)

    # saving image with bounding boxes as intermediate results and for displaying
    # on the client side
    print_("Obtaining intermediate detector visualization...")
    img = img[0,:].cpu().numpy().transpose(1,2,0) / 255
    # img = img[0,:].cpu().numpy().transpose(1,2,0)
    img_name = os.path.basename(img_path)
    create_directory(os.path.join(os.getcwd(), "data", "intermediate_results"))
    savepath = os.path.join(os.getcwd(), "data", "intermediate_results", img_name)
    # case for image with no detections
    if(len(labels[0]) == 0 and os.path.exists(savepath)):
       os.remove(savepath)
       visualize_img(img=img, savefig=True, savepath=savepath)
    # case for image with bounding boxes
    else:
       visualize_bbox(img=img, boxes=boxes[0], labels=labels[0],
                      scores=scores[0], savefig=True, savepath=savepath)

    # extracting the detected person instances and saving them as independent images
    print_("Extracting person detections from image...")
    try:
        detections, centers, scales = DETS_EXTRACTOR(img=img, list_coords=boxes[0])
    except Exception as e:
        detections, centers, scales = [], [], []
    data = {
        "detections": detections,
        "centers": centers,
        "scales": scales
    }
    n_dets = len(detections)
    print_(f"{n_dets} person instances have been detected...")
    det_paths = []

    create_directory(os.path.join(os.getcwd(), "data", "final_results", "detection"))
    for i, det in enumerate(detections):
        det_name = img_name.split(".")[0] + f"_det_{i}." + img_name.split(".")[1]
        det_path = os.path.join(os.getcwd(), "data", "final_results", "detection", det_name)
        det_paths.append(det_path)
        visualize_img(img=det.transpose(1,2,0), savefig=True, savepath=det_path)

    return savepath, det_paths, data


@log_function
def bbox_filtering(predictions, filter_=1, thr=0.6):
    """
    Filtering predicitions in order to keep only the relevant bounding boxes #
    (people in our particular case)

    Args:
    -----
    predictions: list
        list containign a dictionary with all predicted bboxes, labels and scores:
            bbox: numpy array
                Array of shape (N, 4) where N is the number of boxes detected.
                The 4 corresponds to x_min, y_min, x_max, y_max
            label: numpy array
                Array containing the ID for the predicted labels
            scores: numpy array
                Array containing the prediction confident scores
    filter: list
        list containing label indices that we wnat to keep
    thr: float
        score threshold for considering a bounding box
    """

    filtered_bbox, filtered_labels, filtered_scores = [], [], []
    for pred in predictions:
        bbox, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
        cur_bbox, cur_labels, cur_scores = [], [], []
        for i, _ in enumerate(labels):
            if(labels[i] == filter_ and scores[i] > thr):
                aux = bbox[i].cpu().detach().numpy()
                reshaped_bbox = [aux[0], aux[1], aux[2], aux[3]]
                cur_bbox.append(reshaped_bbox)
                # cur_bbox.append(bbox[i].cpu().detach().numpy())
                cur_labels.append(labels[i])
                cur_scores.append(scores[i])
        # if(len(cur_bbox) == 0):
            # continue
        filtered_bbox.append(cur_bbox)
        filtered_labels.append(cur_labels)
        filtered_scores.append(cur_scores)


    return filtered_bbox, filtered_labels, filtered_scores


#
