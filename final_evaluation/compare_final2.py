#import threading
import multiprocessing
import numpy as np

from .compare_final2_compoelem import eval_single_combination as eval_single_combination_compoelem
# from .compare_final2_traditional import eval_single_combination as eval_single_combination_traditional
# from .compare_final2_deep import eval_single_combination as eval_single_combination_deep

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

experiments_combined_resnet = [
    # {
    #     "experiment_name":"combined resnet eucl tuned",

    #     "norm_method":"norm_by_global_action",
    #     "sort_method_name":"hr_additional_desc",

    #     "correction_angle":50,
    #     "cone_opening_angle":70,
    #     "cone_scale_factor":5,
    #     "cone_base_scale_factor":2.5,
    #     "filter_threshold": 150,

    #     "poseline_fallback":True,
    #     "bisection_fallback":False,
    #     "glac_fallback":True,

    #     "compare_other":"resnet50_eucl",
    # },
    # {
    #     "experiment_name":"combined resnet eucl tuned",

    #     "norm_method":"norm_by_global_action",
    #     "sort_method_name":"hr_combi3_desc",

    #     "correction_angle":50,
    #     "cone_opening_angle":70,
    #     "cone_scale_factor":5,
    #     "cone_base_scale_factor":2.5,
    #     "filter_threshold": 150,

    #     "poseline_fallback":True,
    #     "bisection_fallback":False,
    #     "glac_fallback":True,

    #     "compare_other":"resnet50_eucl",
    # },
    # {
    #     "experiment_name":"combined resnet eucl tuned",

    #     "norm_method":"norm_by_global_action",
    #     "sort_method_name":"hr_combi4_desc",

    #     "correction_angle":50,
    #     "cone_opening_angle":70,
    #     "cone_scale_factor":5,
    #     "cone_base_scale_factor":2.5,
    #     "filter_threshold": 150,

    #     "poseline_fallback":True,
    #     "bisection_fallback":False,
    #     "glac_fallback":True,

    #     "compare_other":"resnet50_eucl",
    # },
    # {
    #     "experiment_name":"combined resnet eucl tuned",

    #     "norm_method":"norm_by_global_action",
    #     "sort_method_name":"combi1_asc",

    #     "correction_angle":50,
    #     "cone_opening_angle":70,
    #     "cone_scale_factor":5,
    #     "cone_base_scale_factor":2.5,
    #     "filter_threshold": 150,

    #     "poseline_fallback":True,
    #     "bisection_fallback":False,
    #     "glac_fallback":True,

    #     "compare_other":"resnet50_eucl",
    # },
    # {
    #     "experiment_name":"combined resnet eucl tuned",

    #     "norm_method":"norm_by_global_action",
    #     "sort_method_name":"combi2_asc",

    #     "correction_angle":50,
    #     "cone_opening_angle":70,
    #     "cone_scale_factor":5,
    #     "cone_base_scale_factor":2.5,
    #     "filter_threshold": 150,

    #     "poseline_fallback":True,
    #     "bisection_fallback":False,
    #     "glac_fallback":True,

    #     "compare_other":"resnet50_eucl",
    # },
    {
        "experiment_name":"combined resnet ncos tuned",

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

        "compare_other":"resnet50_cos",
    },
    {
        "experiment_name":"combined resnet ncos tuned",

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

        "compare_other":"resnet50_cos",
    },
    {
        "experiment_name":"combined resnet ncos tuned",

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

        "compare_other":"resnet50_cos",
    },
    {
        "experiment_name":"combined resnet ncos tuned",

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

        "compare_other":"resnet50_cos",
    },
    {
        "experiment_name":"combined resnet ncos tuned",

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

        "compare_other":"resnet50_cos",
    },
]

experiments_combined_resnet_weighted_additional = [
    {
        "experiment_name":"weighted combined resnet eucl tuned",

        "norm_method":"norm_by_global_action",
        "sort_method_name": sort,

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "additional_feature_weight": weight,
        "compare_other":"resnet50_eucl",
    }
    for weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95][::-1]
    #for sort in ["combi1_asc", "combi2_asc", "hr_combi3_desc", "hr_combi4_desc"]
    for sort in ["combi1_asc", "combi2_asc", "hr_combi3_desc", "hr_combi4_desc"]
]
#
# experiments_combined_vgg_weighted_additional = [
#     {
#         "experiment_name":"weighted combined vgg19 ncos tuned",

#         "norm_method":"norm_by_global_action",
#         "sort_method_name": sort,

#         "correction_angle":50,
#         "cone_opening_angle":70,
#         "cone_scale_factor":5,
#         "cone_base_scale_factor":2.5,
#         "filter_threshold": 150,

#         "poseline_fallback":True,
#         "bisection_fallback":False,
#         "glac_fallback":True,

#         "additional_feature_weight": weight,
#         "compare_other":"vgg19_ncos",
#     }
#     for weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
#     for sort in ["combi1_asc", "combi2_asc"]
# ]
experiments_combined_traditional_weighted_additional = [
    {
        "experiment_name":"weighted combined traditional ncos tuned",

        "norm_method":"norm_by_global_action",
        "sort_method_name": sort,

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold": 150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,

        "additional_feature_weight": weight,
        "compare_other": other,
    }
    #for weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for weight in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for sort in ["combi1_asc", "combi2_asc"]
    for other in ["sift_bfm1","orb_bfm1","brief_bfm1"]
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

experiments_traditional = [
    {
        "experiment_name":"traditional baseline",#misstake: "combined tuned SIFT",
        "compare_method_name":"compare_briefBFMatcher1",
        "feature_key":"brief",
    },
    {
        "experiment_name":"traditional baseline",
        "compare_method_name":"compare_briefBFMatcher2",
        "feature_key":"brief",
    },
    {
        "experiment_name":"traditional baseline",
        "compare_method_name":"compare_orbBFMatcher1",
        "feature_key":"orb",
    },
    {
        "experiment_name":"traditional baseline",
        "compare_method_name":"compare_orbBFMatcher2",
        "feature_key":"orb",
    },
    {
        "experiment_name":"traditional baseline",
        "compare_method_name":"compare_siftBFMatcher1",
        "feature_key":"sift",
    },
    {
        "experiment_name":"traditional baseline",
        "compare_method_name":"compare_siftBFMatcher2",
        "feature_key":"sift",
    },
]


experiments_deep = [
    {
        "experiment_name":"deep baseline",
        "compare_method_name":"eucl_dist_flatten",
        "feature_key":"imageNet_vgg19_bn_features",
    },
    {
        "experiment_name":"deep baseline",
        "compare_method_name":"negative_cosine_dist_flatten",
        "feature_key":"imageNet_vgg19_bn_features",
    },
    {
        "experiment_name":"deep baseline",
        "compare_method_name":"eucl_dist_flatten",
        "feature_key":"places365_resnet50_feature_noFC",
    },
    {
        "experiment_name":"deep baseline",
        "compare_method_name":"negative_cosine_dist_flatten",
        "feature_key":"places365_resnet50_feature_noFC",
    },
]

#for exp in experiments_deep:
#    print("exp",exp)
#    eval_single_combination_deep(exp)



second_grid_search = [ #step 2 evaluation bbox norm, other is in seperate compare files
    {
        "experiment_name":"gridsearch 2 - pl,norm dependency",

        "norm_method": norm,
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold": th*1000 if norm == "none" else th, #fix1,fix2

        "poseline_fallback":pl_fb,
        "bisection_fallback":False,
        "glac_fallback":False,
    }
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.45, 0.50]
    for pl_fb in [True, False]
    #for norm in ["minmax_norm_by_imgrect", "minmax_norm_by_bbox", "none"]
    for norm in ["minmax_norm_by_imgrect", "minmax_norm_by_bbox"] #fix 2
    #for norm in ["none"] #disable others for the fix
]

#eval_single_combination_traditional(experiments_traditional[0])


final_grid_search_results = [
    # untuned
    {
        "experiment_name":"BASELINE ICC+ U AR + fix precision curve",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":20,
        "cone_opening_angle":80,
        "cone_scale_factor":10,
        "cone_base_scale_factor":0,
        "filter_threshold":150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,
    },
    # tuned
    {
        "experiment_name":"BASELINE ICC+ T AR + fix precision curve",

        "norm_method":"norm_by_global_action",
        "sort_method_name":"hr_nmd_desc",

        "correction_angle":50,
        "cone_opening_angle":70,
        "cone_scale_factor":5,
        "cone_base_scale_factor":2.5,
        "filter_threshold":150,

        "poseline_fallback":True,
        "bisection_fallback":False,
        "glac_fallback":True,
    },
]

def main():
    print("starting pool")
    p = multiprocessing.Pool()
    print("pool started")
    #p.map(eval_single_combination_compoelem, experiments[0:8])
    #p.map(eval_single_combination_compoelem, experiments[8:len(experiments)])
    #p.map(eval_single_combination_compoelem, experiments[15:len(experiments)])
    #p.map(eval_single_combination_compoelem, experiments[25:27])
    #p.map(eval_single_combination_compoelem, [experiments[19]]) #glac fallback
    #p.map(eval_single_combination_compoelem, experiments2_fbFalse) #glac fallback
    #p.map(eval_single_combination_compoelem, experiments2_fbTrue) #glac fallback
    #p.map(eval_single_combination_compoelem, experiments3_fix) #glac fallback
    #p.map(eval_single_combination_compoelem, experiments_combined_vgg19)
    #p.map(eval_single_combination_compoelem, experiments_combined_sift)
    #p.map(eval_single_combination_traditional, experiments_traditional[0:1])
    #p.map(eval_single_combination_deep, experiments_deep)
    #p.map(eval_single_combination_compoelem, second_grid_search[0:len(second_grid_search)//2]) # laptop
    #p.map(eval_single_combination_compoelem, second_grid_search[len(second_grid_search)//2:len(second_grid_search)]) # lab 
    #p.map(eval_single_combination_compoelem, experiments_combined_resnet)
    #p.map(eval_single_combination_compoelem, experiments_combined_resnet_weighted_additional)
    #p.map(eval_single_combination_compoelem, experiments_combined_vgg_weighted_additional)
    p.map(eval_single_combination_compoelem, experiments_combined_traditional_weighted_additional)
    print("map done")
    p.close()
    print("closed")
    p.join()
    print("joined")

if __name__ == '__main__':
    main()
