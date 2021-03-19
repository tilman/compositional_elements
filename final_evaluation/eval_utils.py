import numpy as np
import pandas as pd

def score_retrievals(label, retrievals):
    """
    Evaluating the current retrieval experiment
    Args:
    -----
    label: string
        label corresponding to the query
    retrivals: list
        list of strings containing the ranked labels corresponding to the retrievals
    tot_labels: integer
        number of images with the current label. We need this to compute recalls
    """
    # retrievals = retrievals[1:] # we do not account rank-0 since it's self-retrieval
    relevant_mask = np.array([1 if r==label else 0 for r in retrievals])
    num_relevant_retrievals = np.sum(relevant_mask)
    if(num_relevant_retrievals == 0):
        print(label)
        metrics = {
            "label": label,
            "p@1": -1,
            "p@5": -1,
            "p@10": -1,
            "p@50": -1,
            "p@rel": -1,
            "mAP": -1,
            "r@1": -1,
            "r@5": -1,
            "r@10": -1,
            "r@50": -1,
            "r@rel": -1,
            "mAR": -1
        }
        return metrics
    # computing precision based metrics
    precision_at_rank = np.cumsum(relevant_mask) / np.arange(1, len(relevant_mask) + 1)
    precision_at_1 = precision_at_rank[0]
    precision_at_5 = precision_at_rank[4]
    precision_at_10 = precision_at_rank[9]
    precision_at_50 = precision_at_rank[49]
    precision_at_rel = precision_at_rank[num_relevant_retrievals - 1] #precision at rellevant retrievals
    average_precision = np.sum(precision_at_rank * relevant_mask) / num_relevant_retrievals
    # computing recall based metrics
    recall_at_rank = np.cumsum(relevant_mask) / num_relevant_retrievals
    recall_at_1 = recall_at_rank[0]
    recall_at_5 = recall_at_rank[4]
    recall_at_10 = recall_at_rank[9]
    recall_at_50 = recall_at_rank[49]
    recall_at_rel = recall_at_rank[num_relevant_retrievals - 1] #recall at rellevant retrievals
    average_recall = np.sum(recall_at_rank * relevant_mask) / num_relevant_retrievals
    metrics = {
        "label": label,
        "p@1": precision_at_1,
        "p@5": precision_at_5,
        "p@10": precision_at_10,
        "p@50": precision_at_50,
        "p@rel": precision_at_rel,
        "mAP": average_precision,
        "r@1": recall_at_1,
        "r@5": recall_at_5,
        "r@10": recall_at_10,
        "r@50": recall_at_50,
        "r@rel": recall_at_rel,
        "mAR": average_recall
    }
    return metrics

def get_eval_dataframe(res_metrics):
    avgerave_metrics = {}
    for metricKey in res_metrics.keys():
        if metricKey != "label":
            if metricKey not in avgerave_metrics:
                avgerave_metrics[metricKey] = {}
            total_list = []
            for label in res_metrics[metricKey].keys():
                avgerave_metrics[metricKey][label] = np.mean(res_metrics[metricKey][label]) # mean for each class
                total_list.append(res_metrics[metricKey][label])
            avgerave_metrics[metricKey]["total (mean)"] = np.mean(list(avgerave_metrics[metricKey].values())) # mean of all classes means
            avgerave_metrics[metricKey]["total (w. mean)"] = np.mean(np.array(total_list).flatten()) # mean of all values regardless of class (-> the same as class mean weighted by amount of datapoints in class)
    return pd.DataFrame(avgerave_metrics)
