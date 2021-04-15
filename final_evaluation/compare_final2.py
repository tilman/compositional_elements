#import threading
import multiprocessing
import numpy as np

from .compare_final2_compoelem import eval_single_combination

# allowed values:
# norm_method: minmax_norm_by_imgrect, minmax_norm_by_bbox, norm_by_global_action, none, 
# sort_method_name: cr_desc, nmd_desc, hr_nmd_desc
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
        "sort_method_name":"hr_nmd_desc",

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
        "sort_method_name":"hr_nmd_desc",

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
        "sort_method_name":"hr_nmd_desc",

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
        "sort_method_name":"hr_nmd_desc",

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
experiments2_fbFalse2 = [
    {
        "experiment_name":"step 2 bbox norm setup",

        "norm_method":"minmax_norm_by_bbox",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 1.1,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "experiment_name":"step 2 bbox norm setup",

        "norm_method":"minmax_norm_by_bbox",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 2,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    },
    {
        "experiment_name":"step 2 bbox norm setup",

        "norm_method":"minmax_norm_by_bbox",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    }
]
experiments3_fix = [
    {
        "experiment_name":"step 2 bbox norm setup",

        "norm_method":"minmax_norm_by_bbox",
        "sort_method_name":"cr_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 0.15,

        "poseline_fallback":False,
        "bisection_fallback":False,
        "glac_fallback":False,
    }
]

experiments_combined_vgg19 = [
    
    {
        "experiment_name":"combined vgg19 baseline",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,
    },
    {
        "experiment_name":"combined vgg19",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_additional_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi3_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi4_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi1_asc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi2_asc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },

##### tuned #############
    
    {
        "experiment_name":"combined vgg19 tuned baseline",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,
    },
    {
        "experiment_name":"combined vgg19 tuned",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_additional_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19 tuned",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi3_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19 tuned",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi4_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19 tuned",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi1_asc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
    {
        "experiment_name":"combined vgg19 tuned",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi2_asc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"vgg19_ncos",
    },
]


experiments_combined_sift = [
    
    {
        "experiment_name":"combined SIFT baseline",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,
    },
    {
        "experiment_name":"combined SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_additional_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi3_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi4_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi1_asc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi2_asc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    # +tuned features  ###########################################
    
    {
        "experiment_name":"combined tuned SIFT baseline",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,
    },
    {
        "experiment_name":"combined tuned SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_additional_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined tuned SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi3_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined tuned SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_combi4_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined tuned SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi1_asc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
    {
        "experiment_name":"combined tuned SIFT",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"combi2_asc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "compare_other":"sift_bfm1",
    },
]


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
    #p.map(eval_single_combination, experiments2_fbTrue) #glac fallback
    #p.map(eval_single_combination, experiments3_fix) #glac fallback
    #p.map(eval_single_combination, experiments_combined_vgg19)
    p.map(eval_single_combination, experiments_combined_sift)
    #p.map(eval_single_combination, experiments_combined_vgg19)
    print("map done")
    p.close()
    print("closed")
    p.join()
    print("joined")

if __name__ == '__main__':
    main()
