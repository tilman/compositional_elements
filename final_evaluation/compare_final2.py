import threading
import numpy as np

from .compare_final2_compoelem import eval_single_combination

# allowed values:
# norm_method: minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action, none, 
# sort_method_name: cr_desc, nmd_desc, lexsort_hr_nmd

# order of the arguments matter!!!
experiments = [
    # BASELINE
    {
        "norm_method":"none",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },


    # vs all norm methods:
    {
        "norm_method":"minmax_norm_by_imgrect",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "norm_method":"minmax_norm_by_bbox",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "norm_method":"norm_by_global_action",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },


    # vs all sort methods:
    {
        "norm_method":"none",
        "sort_method_name":"nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "norm_method":"none",
        "sort_method_name":"lexsort_hr_nmd",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },

    # vs conebase:
    {
        "norm_method":"none",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0.5,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "norm_method":"none",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":1,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "norm_method":"none",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":1.5,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },


    # vs fallbacks:
    {
        "norm_method":"none",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "norm_method":"none",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":True,
        "glac_fallback":False,
    },
    {
        "norm_method":"none",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":True,
    },
]


class myThread (threading.Thread):
   def __init__(self, args):
      threading.Thread.__init__(self)
      self.args = args
   def run(self):
      print ("Starting " + self.name)
      eval_single_combination(**self.args)
      print ("Exiting " + self.name)

# Create new threads

threads = []

for exp in experiments[0:8]:
    threads.append(myThread(exp))

# Start Threads
for j in range(0, len(threads)):
    threads[j].start()

# Join Threads
for j in range(0, len(threads)):
    threads[j].join()
print ("Exiting Main Thread")