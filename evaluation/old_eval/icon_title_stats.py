# call this script with `python -m evaluation.evaluate_poselines_globalaction`
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean, size
from compoelem.types import PoseLine
import os
import sys
from typing import Any, Sequence, Tuple, cast
import numpy as np
import pickle
import time
from compoelem.config import config

import cv2
from tqdm import tqdm
# from compoelem.detect.openpose.lib.utils.common import draw_humans
DATASTORE_FILE = os.path.join(os.path.dirname(__file__), "./datastore_icon_title_2.pkl")
datastore = pickle.load(open(DATASTORE_FILE, "rb"))

counts = {}
for key in datastore.keys():
    className = datastore[key]["row"]["class"]
    if className in counts:
        counts[className] = counts[className] + 1
    else:
        counts[className] = 1
print(counts)