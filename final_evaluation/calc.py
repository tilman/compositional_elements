import os
import numpy as np
import pickle

from tqdm.std import tqdm

from . import calc_compoelem
from . import calc_sift
from . import calc_imageNet_vgg19_bn_features_featuremaps
from . import calc_places365_resnet50_feature_noFC_featuremaps

#dataset_cleaned_extended_balanced = ceb_dataset

DATASTORE_FILE = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem/final_evaluation/combined_datastore_ceb_dataset.pkl" 
DATASET_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dataset_cleaned_extended_balanced"

try:
    datastore = pickle.load(open(DATASTORE_FILE, "rb"))
except FileNotFoundError as e:
    datastore = {}

dataset = np.array([[(className, img) for img in os.listdir( DATASET_ROOT+'/'+className)] for className in os.listdir( DATASET_ROOT )]).reshape(-1,2)
classes, counts = np.unique(dataset[:,0], return_counts=True)
print("total count:",len(dataset)," per class counts",list(zip(classes, counts)))

for className, imgName in tqdm(dataset, total=len(dataset)):
    filename = DATASET_ROOT+'/'+className+'/'+imgName
    key = className+'_'+imgName
    changed = False
    if key not in datastore:
        datastore[key] = {"className":className, "imgName":imgName}
        changed = True
    if "compoelem" not in datastore[key]:
        datastore[key]["compoelem"] = calc_compoelem.precompute(filename)
        changed = True
    if "imageNet_vgg19_bn_features" not in datastore[key]:
        datastore[key]["imageNet_vgg19_bn_features"] = calc_imageNet_vgg19_bn_features_featuremaps.precompute(filename)
        changed = True
    if "places365_resnet50_feature_noFC" not in datastore[key]:
        print("calc resnet")
        datastore[key]["places365_resnet50_feature_noFC"] = calc_places365_resnet50_feature_noFC_featuremaps.precompute(filename)
        changed = True
    # if "sift" not in datastore[key]:
    datastore[key]["sift"] = calc_sift.precompute(filename)
    changed = True
    # if changed:
    #     pickle.dump(datastore, open(DATASTORE_FILE, "wb"))
pickle.dump(datastore, open(DATASTORE_FILE, "wb"))
print("output length", len(datastore.keys()))
