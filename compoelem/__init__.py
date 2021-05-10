__pdoc__ = { 
    # "compoelem.detect" : False,
    # "compoelem.detect.converter" : False, # not working for pdoc because of missing dependency: compoelem.detect.openpose.lib.utils.common
    "compoelem.detect.openpose_wrapper" : False, # not working for pdoc because of missing dependency: compoelem.detect.openpose.lib.utils.common
    # "compoelem.visualize.visualize" : False,
    # "compoelem.compare.normalize" : False,
    "compoelem.detect.openpose" : False, 
    "compoelem.detect.openpose.lib" : False, 
    # "compoelem.detect.openpose.lib.utils" : True, 
    # "compoelem.interactive_demo" : False, 
    # "compoelem.detect.hrnet" : False,
    # "lib" : False,
    # "lib.neural_nets" : False,
}
# from . import detect
from .generate import *
# from . import visualize
version='0.1.1'