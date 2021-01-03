import os
from pickle import GLOBAL
import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from compositional_elements.detect.transforms import TransformDetection
from compositional_elements.detect.bbox_utils import bbox_filtering, bbox_nms

# from compositional_elements.detect.lib.transforms import TransformDetection
# class FasterRCNN:
model = None
extractor = None
def setup():
    global model
    if(model is None):
            
        print("Init faster rcnn")
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model.eval()
        # loading pretrained weights
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "coco_faster_rcnn.pth")
        print(f"    Loading: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(checkpoint)
        model = model.eval()
    # intiializing object for extracting detections
    global extractor
    if(extractor is None):
        extractor = TransformDetection(det_width=192, det_height=256)
    return model, extractor

def get_person_boundingboxes(img_path: str):
    model, extractor = setup()
    # model, extractor
    # loading image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = torch.Tensor(img.transpose(2,0,1)[np.newaxis,:])

    # forward pass through person detector
    print("pass image to Faster-RCNN")
    outputs = model(img / 255)
    boxes, labels, scores = bbox_filtering(outputs, label_filter=1, score_threshold=0.4)
    boxes, labels, scores = bbox_nms(boxes, labels, scores, nms_thr=0.5)
    # saving image with bounding boxes as intermediate results and for displaying
    # on the client side
    print("Obtaining intermediate detector visualization...")
    img = img[0,:].cpu().numpy().transpose(1,2,0) / 255
    # img = img[0,:].cpu().numpy().transpose(1,2,0)
    # img_name = os.path.basename(img_path)
    # create_directory(os.path.join(os.getcwd(), "data", "intermediate_results"))
    # savepath = os.path.join(os.getcwd(), "data", "intermediate_results", img_name)

    # extracting the detected person instances and saving them as independent images
    print("Extracting person detections from image...")
    try:
        detections, centers, scales = extractor(img=img, list_coords=boxes[0])
    except Exception as e:
        detections, centers, scales = [], [], []
    data = {
        "detections": detections,
        "centers": centers,
        "scales": scales,
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
    }
    n_dets = len(detections)
    print(f"{n_dets} person instances have been detected...")
    # det_paths = []
    # final_results_dir = os.path.join(os.getcwd(), "data", "final_results", "detection")
    # os.makedirs(final_results_dir, exist_ok=True)
    # for i, det in enumerate(detections):
    #     det_name = img_name.split(".")[0] + f"_det_{i}." + img_name.split(".")[1]
    #     det_path = os.path.join(os.getcwd(), "data", "final_results", "detection", det_name)
    #     det_paths.append(det_path)
    return data