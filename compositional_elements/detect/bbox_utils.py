def bbox_filtering(predictions, label_filter=1, score_threshold=0.6):
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
    label_filter: list
        list containing label indices that we wnat to keep
    score_threshold: float
        score threshold for considering a bounding box
    """

    # import pdb.set_trace()
    filtered_bbox, filtered_labels, filtered_scores = [], [], []
    for pred in predictions:
        bbox, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
        cur_bbox, cur_labels, cur_scores = [], [], []
        for i, _ in enumerate(labels):
            if(labels[i] == label_filter and scores[i] > score_threshold):
                aux = bbox[i].cpu().detach().numpy()
                reshaped_bbox = [aux[0], aux[1], aux[2], aux[3]]
                cur_bbox.append(reshaped_bbox)
                # cur_bbox.append(bbox[i].cpu().detach().numpy())
                cur_labels.append(labels[i].cpu().detach().numpy())
                cur_scores.append(scores[i].cpu().detach().numpy())
        # if(len(cur_bbox) == 0):
            # continue
        filtered_bbox.append(cur_bbox)
        filtered_labels.append(cur_labels)
        filtered_scores.append(cur_scores)


    return filtered_bbox, filtered_labels, filtered_scores


def bbox_nms(boxes, labels, scores, nms_thr=0.5):
    """
    Applying Non-maximum suppresion to remove redundant bounding boxes
    Args:
    -----
    boxes: list
        List of shape (N, 4) where N is the number of boxes detected.
        The 4 corresponds to x_min, y_min, x_max, y_max
    labels: list
        List containing the ID for the predicted labels
    scores: list
        List containing the prediction confident scores
    nms_thr: float
        threshold used for the NMS procedure
    """

    # import pdb
    # pdb.set_trace()
    boxes_, labels_, scores_ = [], [], []
    for i in range(len(boxes)):
        cur_boxes = np.array(boxes[i])
        cur_labels = np.array(labels[i])
        cur_scores = np.array(scores[i])
        idx = torchvision.ops.nms(boxes = torch.from_numpy(cur_boxes),
                                  scores = torch.from_numpy(cur_scores),
                                  iou_threshold = nms_thr)

        cur_boxes = np.array([cur_boxes[i] for i in idx])
        cur_labels = np.array([cur_labels[i] for i in idx])
        cur_scores = np.array([cur_scores[i] for i in idx])
        boxes_.append(cur_boxes)
        labels_.append(cur_labels)
        scores_.append(cur_scores)

    return np.array(boxes_), np.array(labels_), np.array(scores_)