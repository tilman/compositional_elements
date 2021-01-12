import argparse
from compoelem.detect.openpose.lib.utils.common import Human
from typing import Sequence
import torch
import torch.nn as nn

from . import converter
from .openpose.lib.network.rtpose_vgg import get_model
from .openpose.evaluate.coco_eval import get_outputs
from .openpose.lib.utils.paf_to_pose import paf_to_pose_cpp
from .openpose.lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./compoelem/detect/openpose/experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='./compoelem/detect/openpose/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()


# args = {
#     "cfg":'./compoelem/detect/openpose/experiments/vgg19_368x368_sgd.yaml',
#     "opts":[],
#     "weight":'./compoelem/detect/openpose/pose_model.pth',
# }
# update config file
update_config(cfg, args)

model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model = nn.DataParallel(model)
model.float()
model.eval()

def get_poses(img: Sequence[Sequence[float]]) -> Sequence[Human]:
    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(img, model,  'rtpose')
    # print(im_scale)
    humans = paf_to_pose_cpp(heatmap, paf, cfg)
    return humans