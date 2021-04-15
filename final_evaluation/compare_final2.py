#import threading
import multiprocessing
import numpy as np

from .compare_final2_compoelem import eval_single_combination

# allowed values:
# norm_method: minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action, none, 
# sort_method_name: cr_desc, nmd_desc, lexsort_hr_nmd
# order of the arguments matter!!!

# First step experiments:
experiments = [
    # BASELINE
    {
        "experiment_name":"BASELINE",

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
        "experiment_name":"vs. all norm methods",

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
        "experiment_name":"vs. all norm methods",

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
        "experiment_name":"vs. all norm methods",

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
        "experiment_name":"vs. all sort methods",

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
        "experiment_name":"vs. all sort methods",

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
        "experiment_name":"vs. conebase",

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
        "experiment_name":"vs. conebase",

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
        "experiment_name":"vs. conebase",

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
        "experiment_name":"vs. fallbacks",

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
        "experiment_name":"vs. fallbacks",

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
        "experiment_name":"vs. fallbacks",

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


    ####### norm glac (norm_by_global_action) dependend configurations #########

    # vs conebase:
    {
        "experiment_name":"vs. conebase (norm glac)",

        "norm_method":"norm_by_global_action",
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
        "experiment_name":"vs. conebase (norm glac)",

        "norm_method":"norm_by_global_action",
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
        "experiment_name":"vs. conebase (norm glac)",

        "norm_method":"norm_by_global_action",
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
    {
        "experiment_name":"vs. conebase (norm glac)",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":2,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "experiment_name":"vs. conebase (norm glac)",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":2.5,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "experiment_name":"vs. conebase (norm glac)",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":3,
        "filter_threshold":150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },


    # vs fallbacks:
    {
        "experiment_name":"vs. fallbacks (norm glac)",

        "norm_method":"norm_by_global_action",
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
        "experiment_name":"vs. fallbacks (norm glac)",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,
    },

    ####### poseline fallback dependend configurations #########
    {
        "experiment_name":"vs. all norm methods (pose fb dependend)",

        "norm_method":"minmax_norm_by_imgrect",
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
        "experiment_name":"vs. all norm methods (pose fb dependend)",

        "norm_method":"minmax_norm_by_bbox",
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
        "experiment_name":"vs. all norm methods (pose fb dependend)",

        "norm_method":"norm_by_global_action",
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


    # vs all sort methods:
    {
        "experiment_name":"vs. all sort methods (pose fb dependend)",

        "norm_method":"none",
        "sort_method_name":"nmd_desc",

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
        "experiment_name":"vs. all sort methods (pose fb dependend)",

        "norm_method":"none",
        "sort_method_name":"lexsort_hr_nmd",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":False,
    },

    # idx 25
    ####### with fixed thresholds for imgrect minmax norm #########
    # vs all norm methods:
    {
        "experiment_name":"vs. all norm methods",

        "norm_method":"minmax_norm_by_imgrect",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":0.15,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
]


experiments2_fbTrue = [ #step 2 evaluation bbox norm, other is in seperate compare files
    
    {
        "experiment_name":"step 2 bbox norm setup",

        "norm_method":"minmax_norm_by_bbox",
        "sort_method_name":"lexsort_hr_nmd",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": th,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":False,
    }
    for th in [0.20, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
]
experiments2_fbFalse = [ #step 2 evaluation bbox norm, other is in seperate compare files
    
    {
        "experiment_name":"step 2 bbox norm setup",

        "norm_method":"minmax_norm_by_bbox",
        "sort_method_name":"lexsort_hr_nmd",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": th,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    }
    for th in [0.20, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
]


#class myThread (threading.Thread):
#   def __init__(self, args):
#      threading.Thread.__init__(self)
#      self.args = args
#   def run(self):
#      print ("Starting " + self.name)
#      eval_single_combination(**self.args)
#      print ("Exiting " + self.name)

# Create new threads

# threads = []
# 
# for exp in experiments[8:len(experiments)]:
    # threads.append(myThread(exp))
# 
#Start Threads
# for j in range(0, len(threads)):
    # threads[j].start()
# 
#Join Threads
# for j in range(0, len(threads)):
    # threads[j].join()
# print ("Exiting Main Thread")

def main():
    print("starting pool")
    p = multiprocessing.Pool()
    print("pool started")
    #p.map(eval_single_combination, experiments[0:8])
    #p.map(eval_single_combination, experiments[8:len(experiments)])
    #p.map(eval_single_combination, experiments[15:len(experiments)])
    #p.map(eval_single_combination, experiments[25:27])
    #p.map(eval_single_combination, [experiments[19]]) #glac fallback
    #p.map(eval_single_combination, experiments2_fbFalse) #glac fallback
    p.map(eval_single_combination, experiments2_fbTrue) #glac fallback
    print("map done")
    p.close()
    print("closed")
    p.join()
    print("joined")

if __name__ == '__main__':
    main()
