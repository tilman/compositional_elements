from typing import Sequence
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

print("import local stuff")
from torchopenpose import model
from torchopenpose import util
from torchopenpose.body import Body
from torchopenpose.hand import Hand

def get_poses(img: Sequence[Sequence[float]]):
    body_estimation = Body('model/body_pose_model.pth')
    candidate, subset = body_estimation(img)
    return candidate, subset
    # canvas = util.draw_bodypose(canvas, candidate, subset)
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    # plt.axis('off')
    # plt.show()
